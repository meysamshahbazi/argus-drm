#include "cudaDraw.h"
#include "cudaAlphaBlend.cuh"


#define MIN(a,b)  (a < b ? a : b)
#define MAX(a,b)  (a > b ? a : b)

template<typename T> inline __device__ __host__ T sqr(T x) 				    { return x*x; }

inline __device__ __host__ float dist2(float x1, float y1, float x2, float y2) { return sqr(x1-x2) + sqr(y1-y2); }
inline __device__ __host__ float dist(float x1, float y1, float x2, float y2)  { return sqrtf(dist2(x1,y1,x2,y2)); }

//----------------------------------------------------------------------------
// Line drawing (find if the distance to the line <= line_width)
// Distance from point to line segment - https://stackoverflow.com/a/1501725
//----------------------------------------------------------------------------
inline __device__ float lineDistanceSquared(float x, float y, float x1, float y1, float x2, float y2) 
{
	const float d = dist2(x1, y1, x2, y2);
	const float t = ((x-x1) * (x2-x1) + (y-y1) * (y2-y1)) / d;
	const float u = MAX(0, MIN(1, t));
	
	return dist2(x, y, x1 + u * (x2 - x1), y1 + u * (y2 - y1));
}


/**
 * @brief 	
 * 
 * @tparam T 
 * @param img 
 * @param imgWidth 
 * @param imgHeight 
 * @param offset_x 
 * @param offset_y 
 * @param x1 
 * @param y1 
 * @param x2 
 * @param y2 
 * @param color 
 * @param line_width2 
 * @return __global__ 
 */
template<typename T>
__global__ void gpuDrawLine( T* img, int imgWidth, int imgHeight, int offset_x, int offset_y, int x1, int y1, int x2, int y2, const float4 color, float line_width2 ) 
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x + offset_x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y + offset_y;

	if( x >= imgWidth || y >= imgHeight || x < 0 || y < 0 )
		return;

	if( lineDistanceSquared(x, y, x1, y1, x2, y2) <= line_width2 )
	{
		const int idx = y * imgWidth + x;
		img[idx] = cudaAlphaBlend(img[idx], color);
	}
}
/**
 * @brief 
 * 
 * @tparam T 
 * @param img 
 * @param imgWidth 
 * @param imgHeight 
 * @param offset_x 
 * @param offset_y 
 * @param x1 
 * @param y1 
 * @param x2 
 * @param y2 
 * @param color_y 
 * @param color_u 
 * @param color_v 
 * @param line_width2 
 * @return __global__ 
 */
template<typename T>
__global__ void gpuDrawLineYUYV( T* img, int imgWidth, int imgHeight, int offset_x, int offset_y, 
int x1, int y1, int x2, int y2, uint8_t color_y,uint8_t color_u,uint8_t color_v, float line_width2 ) 
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x + offset_x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y + offset_y;

	if( x >= imgWidth || y >= imgHeight || x < 0 || y < 0 )
		return;

	if( lineDistanceSquared(x, y, x1, y1, x2, y2) <= line_width2 )
	{
		img[y*2*imgWidth+2*x] = color_y;
		img[y*2*imgWidth+4*(x/2)+1] = color_u;
		img[y*2*imgWidth+4*(x/2)+3] = color_v;
	}
}

// cudaDrawLine
cudaError_t cudaDrawLine( void* input, void* output, size_t width, size_t height, imageFormat format, int x1, int y1, int x2, int y2, const float4& color, float line_width )
{
	if( !input || !output || width == 0 || height == 0 || line_width <= 0 )
		return cudaErrorInvalidValue;
	
	// check for lines < 2 pixels in length
	if( dist(x1,y1,x2,y2) < 2.0 )
	{
		LogWarning(LOG_CUDA "cudaDrawLine() - line has length < 2, skipping (%i,%i) (%i,%i)\n", x1, y1, x2, y2);
		return cudaSuccess;
	}
	
	// if the input and output images are different, copy the input to the output
	// this is because we only launch the kernel in the approximate area of the circle
	if( input != output )
		CUDA(cudaMemcpy(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice));
		
	// find a box around the line
	const int left = MIN(x1,x2) - line_width;
	const int right = MAX(x1,x2) + line_width;
	const int top = MIN(y1,y2) - line_width;
	const int bottom = MAX(y1,y2) + line_width;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(right - left, blockDim.x), iDivUp(bottom - top, blockDim.y));

	#define LAUNCH_DRAW_LINE(type) \
		gpuDrawLine<type><<<gridDim, blockDim>>>((type*)output, width, height, left, top, x1, y1, x2, y2, color, line_width * line_width)
	
	// IMAGE_RGBA8
	LAUNCH_DRAW_LINE(uchar4);

	/*
	if( format == IMAGE_RGB8 )
		LAUNCH_DRAW_LINE(uchar3);
	else  if( format == IMAGE_RGBA8 )
		LAUNCH_DRAW_LINE(uchar4);
	else if( format == IMAGE_RGB32F )
		LAUNCH_DRAW_LINE(float3); 
	else if( format == IMAGE_RGBA32F )
		LAUNCH_DRAW_LINE(float4);
	else if( format == IMAGE_YUYV ) {
		uint8_t color_y = static_cast<uint8_t>(((int)(30 * color.x) + (int)(59 * color.y) + (int)(11 * color.z)) / 100);
		uint8_t color_u = static_cast<uint8_t>(((int)(-17 * color.x) - (int)(33 * color.y) + (int)(50 * color.z) + 12800) / 100);
		uint8_t color_v = static_cast<uint8_t>(((int)(50 * color.x) - (int)(42 * color.y) - (int)(8 * color.z) + 12800) / 100);
		gpuDrawLineYUYV<unsigned char><<<gridDim, blockDim>>>((unsigned char *)output,width,height,left,top,x1,y1,x2,y2,
															color_y,color_u,color_v, line_width*line_width) ;
	}
	else {
		imageFormatErrorMsg(LOG_CUDA, "cudaDrawLine()", format);
		return cudaErrorInvalidValue;
	}*/
		
	return cudaGetLastError();
}

//----------------------------------------------------------------------------
// Rect drawing (a grid of threads is launched over the rect)
//----------------------------------------------------------------------------
template<typename T>
__global__ void gpuDrawRect( T* img, int imgWidth, int imgHeight, int x0, int y0, int boxWidth, int boxHeight, const float4 color ) 
{
	const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	if( box_x >= boxWidth || box_y >= boxHeight )
		return;

	const int x = box_x + x0;
	const int y = box_y + y0;

	if( x >= imgWidth || y >= imgHeight || x < 0 || y < 0 )
		return;

	const int idx = y * imgWidth + x;
	img[idx] = cudaAlphaBlend(img[idx], color);
}

template<typename T>
__global__ void gpuDrawRectYUYV( T* img, int imgWidth, int imgHeight,
int x0, int y0, int boxWidth, int boxHeight, const float4 color ) 
{
	const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	if( box_x >= boxWidth || box_y >= boxHeight )
		return;

	const int x = box_x + x0;
	const int y = box_y + y0;

	if( x >= imgWidth || y >= imgHeight || x < 0 || y < 0 )
		return;

	// const int idx = y * imgWidth + x;
	// img[idx] = cudaAlphaBlend(img[idx], color);
	img[y*2*imgWidth+2*x] = static_cast<unsigned char>(color.x);
	img[y*2*imgWidth+4*(x/2)+1] = static_cast<unsigned char>(color.y);
	img[y*2*imgWidth+4*(x/2)+3] = static_cast<unsigned char>(color.z);
}

// cudaDrawRect
cudaError_t cudaDrawRect( void* input, void* output, size_t width, size_t height, imageFormat format, int left, int top, int right, int bottom, const float4& color, const float4& line_color, float line_width )
{
	if( !input || !output || width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// if the input and output images are different, copy the input to the output
	// this is because we only launch the kernel in the approximate area of the circle
	if( input != output )
		CUDA(cudaMemcpy(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice));
		
	// make sure the coordinates are ordered
	if( left > right )
	{
		const int swap = left;
		left = right;
		right = swap;
	}
	
	if( top > bottom )
	{
		const int swap = top;
		top = bottom;
		bottom = swap;
	}
	
	const int boxWidth = right - left;
	const int boxHeight = bottom - top;
	
	if( boxWidth <= 0 || boxHeight <= 0 )
	{
		LogError(LOG_CUDA "cudaDrawRect() -- rect had width/height <= 0  left=%i top=%i right=%i bottom=%i\n", left, top, right, bottom);
		return cudaErrorInvalidValue;
	}

	// rect fill
	if( color.w > 0 )
	{
		const dim3 blockDim(8, 8);
		const dim3 gridDim(iDivUp(boxWidth,blockDim.x), iDivUp(boxHeight,blockDim.y));
				
		#define LAUNCH_DRAW_RECT(type) \
			gpuDrawRect<type><<<gridDim, blockDim>>>((type*)output, width, height, left, top, boxWidth, boxHeight, color)
		

		//IMAGE_RGBA8
		LAUNCH_DRAW_RECT(uchar4);

/* 
		if( format == IMAGE_RGB8 )
			LAUNCH_DRAW_RECT(uchar3);
		else if( format == IMAGE_RGBA8 )
			LAUNCH_DRAW_RECT(uchar4);
		else if( format == IMAGE_RGB32F )
			LAUNCH_DRAW_RECT(float3); 
		else if( format == IMAGE_RGBA32F )
			LAUNCH_DRAW_RECT(float4);
		else if( format == IMAGE_YUYV ){
			uint8_t color_y = static_cast<uint8_t>(((int)(30 * color.x) + (int)(59 * color.y) + (int)(11 * color.z)) / 100);
			uint8_t color_u = static_cast<uint8_t>(((int)(-17 * color.x) - (int)(33 * color.y) + (int)(50 * color.z) + 12800) / 100);
			uint8_t color_v = static_cast<uint8_t>(((int)(50 * color.x) - (int)(42 * color.y) - (int)(8 * color.z) + 12800) / 100);
			float4 color_yuyv = make_float4(color_y,color_u,color_v,0);
			gpuDrawRectYUYV<<<gridDim, blockDim>>>((unsigned char*)output, width, height, left, top, boxWidth, boxHeight, color_yuyv);
		}
		else {
			imageFormatErrorMsg(LOG_CUDA, "cudaDrawRect()", format);
			return cudaErrorInvalidValue;
		} */
	}
	
	// rect outline
	if( line_color.w > 0 && line_width > 0 )
	{
		int lines[4][4] = {
			{left, top, right, top},
			{right, top, right, bottom},
			{right, bottom, left, bottom},
			{left, bottom, left, top}
		};
		
		for( uint32_t n=0; n < 4; n++ )
			CUDA(cudaDrawLine(output, width, height, format, lines[n][0], lines[n][1], lines[n][2], lines[n][3], line_color, line_width));
	}
	
	return cudaGetLastError();
}



