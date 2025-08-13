/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "cudaFont.h"
#include "cudaVector.h"
#include "cudaOverlay.h"
#include "cudaMappedMemory.h"

// #include "imageIO.h"
#include "filesystem.h"
#include "logging.h"

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

//#define DEBUG_FONT


// Struct for one character to render
struct __align__(16) GlyphCommand
{
	short x;		// x coordinate origin in output image to begin drawing the glyph at 
	short y;		// y coordinate origin in output image to begin drawing the glyph at 
	short u;		// x texture coordinate in the baked font map where the glyph resides
	short v;		// y texture coordinate in the baked font map where the glyph resides 
	short width;	// width of the glyph in pixels
	short height;	// height of the glyph in pixels
};


// adaptFontSize
float adaptFontSize( uint32_t dimension )
{
	const float max_font = 32.0f;
	const float min_font = 28.0f;

	const uint32_t max_dim = 1536;
	const uint32_t min_dim = 768;

	if( dimension > max_dim )
		dimension = max_dim;

	if( dimension < min_dim )
		dimension = min_dim;

	const float dim_ratio = float(dimension - min_dim) / float(max_dim - min_dim);

	return min_font + dim_ratio * (max_font - min_font);
}


// constructor
cudaFont::cudaFont()
{
	mSize = 0.0f;
	
	mCommandCPU = NULL;
	mCommandGPU = NULL;
	mCmdIndex   = 0;

	mFontMapCPU = NULL;
	mFontMapGPU = NULL;

	mRectsCPU   = NULL;
	mRectsGPU   = NULL;
	mRectIndex  = 0;

	mFontMapWidth  = 512;
	mFontMapHeight = 512;
}



// destructor
cudaFont::~cudaFont() {
	if( mRectsCPU != NULL ) {
		CUDA(cudaFreeHost(mRectsCPU));
		
		mRectsCPU = NULL; 
		mRectsGPU = NULL;
	}

	if( mCommandCPU != NULL ) {
		CUDA(cudaFreeHost(mCommandCPU));
		
		mCommandCPU = NULL; 
		mCommandGPU = NULL;
	}

	if( mFontMapCPU != NULL ) {
		CUDA(cudaFreeHost(mFontMapCPU));
		
		mFontMapCPU = NULL; 
		mFontMapGPU = NULL;
	}
}

cudaFont* cudaFont::CreateWithBorder() {
	cudaFont* c = new cudaFont();
	c->init_border();
	return c;
}

// Create
cudaFont* cudaFont::Create( float size ) {
	// default fonts	
	std::vector<std::string> fonts;
	
	fonts.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf");
	fonts.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

	return Create(fonts, size);
}


// Create
cudaFont* cudaFont::Create( const std::vector<std::string>& fonts, float size ) {
	const uint32_t numFonts = fonts.size();

	for( uint32_t n=0; n < numFonts; n++ ) {
		cudaFont* font = Create(fonts[n].c_str(), size);

		if( font != NULL )
			return font;
	}

	return NULL;
}


// Create
cudaFont* cudaFont::Create( const char* font, float size ) {
	// verify parameters
	if( !font )
		return Create(size);

	// create new font
	cudaFont* c = new cudaFont();
	
	if( !c )
		return NULL;
		
	if( !c->init(font, size) ) {
		delete c;
		return NULL;
	}

	return c;
}

bool cudaFont::init_border() {
    std::ifstream gtIfstream("/home/user/data/GlyphInfo.txt");
    std::string gtLine;
    for (int n = 0; n < NumGlyphs; n++) {
        getline(gtIfstream, gtLine);
        std::stringstream gtStream(gtLine);
        std::string element;
        std::vector<int> elements;

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].x = uint16_t(std::atof(element.c_str()));

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].y = uint16_t(std::atof(element.c_str()));  
        
        std::getline(gtStream, element, ',');
        mGlyphInfo[n].width = uint16_t(std::atof(element.c_str()));  

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].height = uint16_t(std::atof(element.c_str()));  

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].xAdvance = float(std::atof(element.c_str()));  

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].xOffset = float(std::atof(element.c_str()));

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].yOffset = float(std::atof(element.c_str()));

    }

	mGlyphInfo[0].height = 0;

    gtIfstream.close();

	const size_t fontMapSize = mFontMapWidth * mFontMapHeight * sizeof(unsigned char);

	if( !cudaAllocMapped((void**)&mFontMapCPU, (void**)&mFontMapGPU, fontMapSize) ) {
			LogError(LOG_CUDA "failed to allocate %zu bytes to store %ix%i font map\n", fontMapSize, mFontMapWidth, mFontMapHeight);
		// free(ttf_buffer_Bold);
		return false;
	}

	if( !cudaAllocMapped((void**)&mFontMapCPU_border, (void**)&mFontMapGPU_border, fontMapSize) ) {
		LogError(LOG_CUDA "failed to allocate %zu bytes to store %ix%i font map\n", fontMapSize, mFontMapWidth, mFontMapHeight);
		return false;
	}

	
	// allocate memory for GPU command buffer	
	if( !cudaAllocMapped(&mCommandCPU, &mCommandGPU, sizeof(GlyphCommand) * MaxCommands) )
		return false;

	// CUDA(cudaMemcpy(mFontMapGPU, font_data, fontMapSize, cudaMemcpyHostToDevice));
    // CUDA(cudaMemcpy(mFontMapGPU_border, font_border_data, fontMapSize, cudaMemcpyHostToDevice));

	std::ifstream fin("/home/user/data/fill_list.bin", std::ios::binary);
    uchar fill_list[fontMapSize];
    fin.read((char*)fill_list, fontMapSize);
	CUDA(cudaMemcpy(mFontMapGPU, fill_list, fontMapSize, cudaMemcpyHostToDevice));
	fin.close();

	std::ifstream fin_b("/home/user/data/border_list.bin", std::ios::binary);
    uchar border_list[fontMapSize];
    fin_b.read((char*)border_list, fontMapSize);
	CUDA(cudaMemcpy(mFontMapGPU_border, border_list, fontMapSize, cudaMemcpyHostToDevice));
	fin_b.close();
	
	// --- init fa:
	std::ifstream gtIfstream2("/home/user/data/fa-GlyphInfo.txt");
    // std::string gtLine;
    for (int n = 0; n < 1883; n++) {
        getline(gtIfstream2, gtLine);
        std::stringstream gtStream(gtLine);
        std::string element;
        // std::vector<int> elements;

        std::getline(gtStream, element, ',');
        fa_GlyphInfo[n].x = uint16_t(std::atof(element.c_str()));

        std::getline(gtStream, element, ',');
        fa_GlyphInfo[n].y = uint16_t(std::atof(element.c_str()));  
        
        std::getline(gtStream, element, ',');
        fa_GlyphInfo[n].width = uint16_t(std::atof(element.c_str()));  

        std::getline(gtStream, element, ',');
        fa_GlyphInfo[n].height = uint16_t(std::atof(element.c_str()));  

        std::getline(gtStream, element, ',');
        fa_GlyphInfo[n].xAdvance = float(std::atof(element.c_str()));  

        std::getline(gtStream, element, ',');
        fa_GlyphInfo[n].xOffset = float(std::atof(element.c_str()));

        std::getline(gtStream, element, ',');
        fa_GlyphInfo[n].yOffset = float(std::atof(element.c_str()));

		std::getline(gtStream, element, ',');
		icon_mp_idx[element] = n; 
		// std::cout << element << "\t";
    }
	std::cout<< std::endl;

    gtIfstream2.close();


	
	const size_t fa_fontMapSize = 1024 * 1024 * sizeof(unsigned char);

	// if( !cudaAllocMapped((void**)&faFontMapCPU, (void**)faFontMapGPU, fa_fontMapSize) ) {
	// 		LogError(LOG_CUDA "failed to allocate %zu bytes to store %ix%i font map\n", fa_fontMapSize, mFontMapWidth, mFontMapHeight);
	// 	// free(ttf_buffer_Bold);
	// 	return false;
	// }

	cudaMalloc(&faFontMapGPU, fa_fontMapSize);
	cudaMalloc(&faFontMapGPU_border, fa_fontMapSize);

	// if( !cudaAllocMapped((void**)&faFontMapCPU_border, (void**)&faFontMapGPU_border, fa_fontMapSize) ) {
	// 	LogError(LOG_CUDA "failed to allocate %zu bytes to store %ix%i font map\n", fa_fontMapSize, mFontMapWidth, mFontMapHeight);
	// 	return false;
	// }

	
	// allocate memory for GPU command buffer	
	if( !cudaAllocMapped(&faCommandCPU, &faCommandGPU, sizeof(GlyphCommand) * FaMaxCommands) ) {
		return false;
	}
		

	std::ifstream finfa("/home/user/data/fa_fill_list.bin", std::ios::binary);
    uchar fa_fill_list[fa_fontMapSize];
    finfa.read((char*)fa_fill_list, fa_fontMapSize);
	CUDA(cudaMemcpy(faFontMapGPU, fa_fill_list, fa_fontMapSize, cudaMemcpyHostToDevice));
	finfa.close();

	std::ifstream fin_bfa("/home/user/data/fa_border_list.bin", std::ios::binary);
    uchar fa_border_list[fa_fontMapSize];
    fin_bfa.read((char*)fa_border_list, fa_fontMapSize);
	
	CUDA(cudaMemcpy(faFontMapGPU_border, fa_border_list, fa_fontMapSize, cudaMemcpyHostToDevice));
	fin_bfa.close();
	return true;
}

// init
bool cudaFont::init( const char* filename, float size ) {
	// validate parameters
	if( !filename )
		return NULL;

	// verify that the font file exists and get its size
	const size_t ttf_size = fileSize(filename);

	if( !ttf_size ) {
		LogError(LOG_CUDA "font doesn't exist or empty file '%s'\n", filename);
 		return false;
	}

	// allocate memory to store the font file
	void* ttf_buffer = malloc(ttf_size);

	if( !ttf_buffer ) {
		LogError(LOG_CUDA "failed to allocate %zu byte buffer for reading '%s'\n", ttf_size, filename);
		return false;
	}

	// open the font file
	FILE* ttf_file = fopen(filename, "rb");

	if( !ttf_file ) {
		LogError(LOG_CUDA "failed to open '%s' for reading\n", filename);
		free(ttf_buffer);
		return false;
	}

	// read the font file
	const size_t ttf_read = fread(ttf_buffer, 1, ttf_size, ttf_file);

	fclose(ttf_file);

	if( ttf_read != ttf_size ) {
		LogError(LOG_CUDA "failed to read contents of '%s'\n", filename);
		LogError(LOG_CUDA "(read %zu bytes, expected %zu bytes)\n", ttf_read, ttf_size);

		free(ttf_buffer);
		return false;
	}

	// buffer that stores the coordinates of the baked glyphs
	stbtt_bakedchar bakeCoords[NumGlyphs];

	// increase the size of the bitmap until all the glyphs fit
	while(true) {
		// allocate memory for the packed font texture (alpha only)
		const size_t fontMapSize = mFontMapWidth * mFontMapHeight * sizeof(unsigned char);

		if( !cudaAllocMapped((void**)&mFontMapCPU, (void**)&mFontMapGPU, fontMapSize) ) {
			LogError(LOG_CUDA "failed to allocate %zu bytes to store %ix%i font map\n", fontMapSize, mFontMapWidth, mFontMapHeight);
			free(ttf_buffer);
			return false;
		}

		// attempt to pack the bitmap
		const int result = stbtt_BakeFontBitmap((uint8_t*)ttf_buffer, 0, size, 
										mFontMapCPU, mFontMapWidth, mFontMapHeight,
									     FirstGlyph, NumGlyphs, bakeCoords);

		if( result == 0 ) {
			LogError(LOG_CUDA "failed to bake font bitmap '%s'\n", filename);
			free(ttf_buffer);
			return false;
		}
		else if( result < 0 ) {
			const int glyphsPacked = -result;

			if( glyphsPacked == NumGlyphs )
			{
				LogVerbose(LOG_CUDA "packed %u glyphs in %ux%u bitmap (font size=%.0fpx)\n", NumGlyphs, mFontMapWidth, mFontMapHeight, size);
				break;
			}

		#ifdef DEBUG_FONT
			LogDebug(LOG_CUDA "fit only %i of %u font glyphs in %ux%u bitmap\n", glyphsPacked, NumGlyphs, mFontMapWidth, mFontMapHeight);
		#endif

			CUDA(cudaFreeHost(mFontMapCPU));
		
			mFontMapCPU = NULL; 
			mFontMapGPU = NULL;

			mFontMapWidth *= 2;
			mFontMapHeight *= 2;

		#ifdef DEBUG_FONT
			LogDebug(LOG_CUDA "attempting to pack font with %ux%u bitmap...\n", mFontMapWidth, mFontMapHeight);
		#endif
			continue;
		}
		else {
		#ifdef DEBUG_FONT
			LogDebug(LOG_CUDA "packed %u glyphs in %ux%u bitmap (font size=%.0fpx)\n", NumGlyphs, mFontMapWidth, mFontMapHeight, size);
		#endif		
			break;
		}
	}

	// free the TTF font data
	free(ttf_buffer);

	// store texture baking coordinates
	for( uint32_t n=0; n < NumGlyphs; n++ ) {
		mGlyphInfo[n].x = bakeCoords[n].x0;
		mGlyphInfo[n].y = bakeCoords[n].y0;

		mGlyphInfo[n].width  = bakeCoords[n].x1 - bakeCoords[n].x0;
		mGlyphInfo[n].height = bakeCoords[n].y1 - bakeCoords[n].y0;

		mGlyphInfo[n].xAdvance = bakeCoords[n].xadvance;
		mGlyphInfo[n].xOffset  = bakeCoords[n].xoff;
		mGlyphInfo[n].yOffset  = bakeCoords[n].yoff;

	#ifdef DEBUG_FONT
		// debug info
		const char c = n + FirstGlyph;
		LogDebug("Glyph %u: '%c' width=%hu height=%hu xOffset=%.0f yOffset=%.0f xAdvance=%0.1f\n",
			n, c, mGlyphInfo[n].width, mGlyphInfo[n].height, mGlyphInfo[n].xOffset, 
			mGlyphInfo[n].yOffset, mGlyphInfo[n].xAdvance);
	#endif	
	}

	// allocate memory for GPU command buffer	
	if( !cudaAllocMapped(&mCommandCPU, &mCommandGPU, sizeof(GlyphCommand) * MaxCommands) )
		return false;
	
	// allocate memory for background rect buffers
	if( !cudaAllocMapped((void**)&mRectsCPU, (void**)&mRectsGPU, sizeof(float4) * MaxCommands) )
		return false;

	mSize = size;
	return true;
}



inline __host__ __device__ float4 alpha_blend( const float4& bg, const float4& fg )
{
	const float alpha = fg.w / 255.0f;
	const float ialph = 1.0f - alpha;
	
	return make_float4(	alpha * fg.x + ialph * bg.x,
						alpha * fg.y + ialph * bg.y,
						alpha * fg.z + ialph * bg.z,
						alpha * fg.w + ialph * bg.w);
} 

inline __host__ __device__ uchar4 alpha_blend( const uchar4& bg, const uchar4& fg )
{
	const uchar alpha = fg.w;
	const uchar ialph = 256 - alpha;
	
	return make_uchar4(	(alpha * fg.x + ialph * bg.x)/256,
						(alpha * fg.y + ialph * bg.y)/256,
						(alpha * fg.z + ialph * bg.z)/256,
						(alpha * fg.w + ialph * bg.w)/256);
} 


__global__ void gpuOverlayTextWithBorder(unsigned char* font, unsigned char* font_border,int fontWidth, GlyphCommand* commands,
						  uchar4* input, uchar4* output, int imgWidth, int imgHeight, uchar4 color, uchar4 color_border) 
{
	const GlyphCommand cmd = commands[blockIdx.x];

	if( threadIdx.x >= cmd.width || threadIdx.y >= cmd.height )
		return;

	const int x = cmd.x + threadIdx.x;
	const int y = cmd.y + threadIdx.y;

	if( x < 0 || y < 0 || x >= imgWidth || y >= imgHeight )
		return;

	const int u = cmd.u + threadIdx.x ;
	const int v = cmd.v + threadIdx.y ;

	const uchar px_glyph = font[v * fontWidth + u];
	const uchar px_glyph_border = font_border[v * fontWidth + u];

	// #define CUDA_FONT_USER_ALPHA_BLEND
	if (px_glyph_border) {
	#ifdef CUDA_FONT_USER_ALPHA_BLEND	
		uchar4 color_border_ = color_border;
		color_border_.w = px_glyph_border;
		output[y * imgWidth + x] = alpha_blend(input[y * imgWidth + x], color_border_ );
	#else 
		output[y * imgWidth + x] = color_border;
	#endif
	}

	if(px_glyph){
	#ifdef CUDA_FONT_USER_ALPHA_BLEND	
		uchar4 color_ = color;
		color_.w = px_glyph;
		output[y * imgWidth + x] = alpha_blend(output[y * imgWidth + x], color_);
	#else 
		output[y * imgWidth + x] = color;
	#endif
	}

}


// cudaOverlayText
cudaError_t cudaOverlayText(unsigned char* font, unsigned char* font_border, const int2& maxGlyphSize, size_t fontMapWidth,
				GlyphCommand* commands, /* GlyphCommand* commands_in, */ size_t numCommands, const uchar4& fontColor, const uchar4& fontColor_out,
				cudaStream_t stream, void* input, void* output, imageFormat format, size_t imgWidth, size_t imgHeight)	
{	
	if( !font || !commands || !input || !output || numCommands == 0 || fontMapWidth == 0 || imgWidth == 0 || imgHeight == 0 )
		return cudaErrorInvalidValue;

	const dim3 block(maxGlyphSize.x, maxGlyphSize.y);
	const dim3 grid(numCommands);

	gpuOverlayTextWithBorder<<<grid, block, 0, stream>>>(font, font_border, fontMapWidth, commands,
		(uchar4*)input, (uchar4*)output, imgWidth, imgHeight, fontColor, fontColor_out);

	return cudaGetLastError();
}

// Overlay
bool cudaFont::OverlayText( void* image, imageFormat format, uint32_t width, uint32_t height, 
							const std::vector< std::pair< std::string, int2 > >& strings, 
							const uchar4& color, const uchar4& color_out,cudaStream_t stream)
{
	const uint32_t numStrings = strings.size();

	if( !image || width == 0 || height == 0 || numStrings == 0 )
		return false;

	// if( format != IMAGE_RGB8 && format != IMAGE_RGBA8 && format != IMAGE_RGB32F && format != IMAGE_RGBA32F && format != IMAGE_ABGR8 && format != IMAGE_YUYV ) {
	if(  format != IMAGE_RGBA8 ) { // we just accept RGBA8!
		LogError(LOG_CUDA "cudaFont::OverlayText() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_CUDA "                           supported formats are:\n");
		LogError(LOG_CUDA "                              * rgba8\n");		

		return false;
	}

	int2 maxGlyphSize = make_int2(0,0);

	int numCommands = 0;
	int maxChars = 0;

	// find the bg rects and total char count
	for( uint32_t s=0; s < numStrings; s++ )
		maxChars += strings[s].first.size();

	// reset the buffer indices if we need the space
	if( mCmdIndex + maxChars >= MaxCommands )
		mCmdIndex = 0;

	// generate glyph commands and bg rects
	for( uint32_t s=0; s < numStrings; s++ )
	{
		const uint32_t numChars = strings[s].first.size();
		
		if( numChars == 0 )
			continue;

		// determine the max 'height' of the string
		int maxHeight = 0;

		for( uint32_t n=0; n < numChars; n++ )
		{
			char c = strings[s].first[n];
			
			if( c < FirstGlyph || c > LastGlyph )
				continue;
			
			c -= FirstGlyph;

			const int yOffset = abs((int)mGlyphInfo[c].yOffset);

			if( maxHeight < yOffset )
				maxHeight = yOffset;
		}

	#ifdef DEBUG_FONT
		LogDebug(LOG_CUDA "max glyph height:  %i\n", maxHeight);
	#endif

		// get the starting position of the string
		int2 pos = strings[s].second;

		if( pos.x < 0 )
			pos.x = 0;

		if( pos.y < 0 )
			pos.y = 0;
		
		// pos.y += maxHeight;

		// make a glyph command for each character
		for( uint32_t n=0; n < numChars; n++ ) {
			char c = strings[s].first[n];
			
			// make sure the character is in range
			if( c < FirstGlyph || c > LastGlyph )
				continue;
			
			c -= FirstGlyph;	// rebase char against glyph 0
			
			// fill the next command
			GlyphCommand* cmd = ((GlyphCommand*)mCommandCPU) + mCmdIndex + numCommands;

			cmd->x = pos.x; // ORGINAL
			// cmd->x = pos.x + mGlyphInfo[c].xOffset;
			cmd->y = pos.y + mGlyphInfo[c].yOffset;
			cmd->u = mGlyphInfo[c].x;// these are start point of char c in font
			cmd->v = mGlyphInfo[c].y;

			cmd->width  = mGlyphInfo[c].width;
			cmd->height = mGlyphInfo[c].height;

			pos.x += mGlyphInfo[c].xAdvance;

			// track the maximum glyph size
			if( maxGlyphSize.x < mGlyphInfo[c].width )
				maxGlyphSize.x = mGlyphInfo[c].width;

			if( maxGlyphSize.y < mGlyphInfo[c].height )
				maxGlyphSize.y = mGlyphInfo[c].height;
			numCommands++;
		}
	}

#ifdef DEBUG_FONT
	LogDebug(LOG_CUDA "max glyph size is %ix%i\n", maxGlyphSize.x, maxGlyphSize.y);
#endif
	CUDA(cudaOverlayText( mFontMapGPU, mFontMapGPU_border, maxGlyphSize, mFontMapWidth,
					((GlyphCommand*)mCommandGPU) + mCmdIndex, 
					numCommands, 
					color,color_out,stream, image, image, format, width, height));

	// advance the buffer indices
	mCmdIndex += numCommands;		   
	return true;
}


bool cudaFont::putIcon( void* image, std::string icon_name, uint32_t width, uint32_t height, 
							int x, int y, const uchar4& color, const uchar4& color_out,cudaStream_t stream)
{

	if( !image || width == 0 || height == 0)
		return false;

	int2 maxGlyphSize = make_int2(0,0);

	int numCommands = 0;
	int maxChars = 0;


	// reset the buffer indices if we need the space
	if( faCmdIndex + maxChars >= FaMaxCommands )
		faCmdIndex = 0;

		// get the starting position of the string
	int2 pos = make_int2(x, y);

	if( pos.x < 0 )
		pos.x = 0;

	if( pos.y < 0 )
		pos.y = 0;
		
	int c = icon_mp_idx[icon_name];
	
	// std::cout << "c is " << c << std::endl;
	
	// fill the next command
	GlyphCommand* cmd = ((GlyphCommand*)faCommandCPU) + faCmdIndex + numCommands;

	cmd->x = pos.x; // ORGINAL
	// cmd->x = pos.x + fa_GlyphInfo[c].xOffset;
	cmd->y = pos.y; //+ fa_GlyphInfo[c].yOffset;
	cmd->u = fa_GlyphInfo[c].x;// these are start point of char c in font
	cmd->v = fa_GlyphInfo[c].y;

	cmd->width  = fa_GlyphInfo[c].width;
	cmd->height = fa_GlyphInfo[c].height;

	// pos.x += fa_GlyphInfo[c].xAdvance;
	// track the maximum glyph size
	if( maxGlyphSize.x < fa_GlyphInfo[c].width )
		maxGlyphSize.x = fa_GlyphInfo[c].width;

	if( maxGlyphSize.y < fa_GlyphInfo[c].height )
		maxGlyphSize.y = fa_GlyphInfo[c].height;
	numCommands++;
		

	// std::cout<< cmd->x << " " << cmd->y <<" " <<  cmd->u << " " << cmd->v << " " << cmd->width  << " " << cmd->height << " "<< numCommands<<  std::endl;
	CUDA(cudaOverlayText( faFontMapGPU, faFontMapGPU_border, maxGlyphSize, 1024,
					((GlyphCommand*)faCommandGPU) + faCmdIndex, 
					numCommands, 
					color,color_out,stream, image, image, IMAGE_RGBA8, width, height));

	// advance the buffer indices
	faCmdIndex += numCommands;		   
	return true;
}


// Overlay
bool cudaFont::OverlayText( void* image, imageFormat format, uint32_t width, uint32_t height, 
					   		const char* str, int x, int y, 
					   		const uchar4& color, const uchar4& color_out, cudaStream_t stream)
{
	if( !str )
		return NULL;
		
	std::vector< std::pair< std::string, int2 > > list;
	
	list.push_back( std::pair< std::string, int2 >( str, make_int2(x,y) ));

	return OverlayText(image, format, width, height, list, color, color_out, stream);
}



// TextExtents

// int2 cudaFont::getWH( const char c,)

int4 cudaFont::TextExtents( const char* str, int x, int y )
{
	if( !str )
		return make_int4(0,0,0,0);

	const size_t numChars = strlen(str);

	// determine the max 'height' of the string
	int maxHeight = 0;

	for( uint32_t n=0; n < numChars; n++ )
	{
		char c = str[n];
		
		if( c < FirstGlyph || c > LastGlyph )
			continue;
		
		c -= FirstGlyph;

		const int yOffset = abs((int)mGlyphInfo[c].yOffset) + mGlyphInfo[c].height;

		if( maxHeight < yOffset )
			maxHeight = yOffset;
	}

	// get the starting position of the string
	int2 pos = make_int2(x,y);

	if( pos.x < 0 )
		pos.x = 0;

	if( pos.y < 0 )
		pos.y = 0;
	
	pos.y += maxHeight;


	// find the extents of the string
	for( uint32_t n=0; n < numChars; n++ ) {
		char c = str[n];
		
		// make sure the character is in range
		if( c < FirstGlyph || c > LastGlyph )
			continue;
		
		c -= FirstGlyph;	// rebase char against glyph 0
		
		// advance the text position
		pos.x += mGlyphInfo[c].xAdvance;
	}

	return make_int4(x, y, pos.x, pos.y);
}
