
#ifndef __CUDA_DRAW_H__
#define __CUDA_DRAW_H__

#include "cudaUtility.h"
#include "imageFormat.h"


// /**
//  * cudaDrawCircle
//  * @ingroup drawing
//  */
// cudaError_t cudaDrawCircle( void* input, void* output, size_t width, size_t height, imageFormat format, 
// 					   int cx, int cy, float radius, const float4& color );
	
// /**
//  * cudaDrawCircle
//  * @ingroup drawing
//  */
// template<typename T> 
// cudaError_t cudaDrawCircle( T* input, T* output, size_t width, size_t height, 
// 				 	   int cx, int cy, float radius, const float4& color )	
// { 
// 	return cudaDrawCircle(input, output, width, height, imageFormatFromType<T>(), cx, cy, radius, color); 
// }	

// /**
//  * cudaDrawCircle (in-place)
//  * @ingroup drawing
//  */
// inline cudaError_t cudaDrawCircle( void* image, size_t width, size_t height, imageFormat format, 
// 							int cx, int cy, float radius, const float4& color )
// {
// 	return cudaDrawCircle(image, image, width, height, format, cx, cy, radius, color);
// }

/**
 * cudaDrawCircle (in-place)
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawCircle( T* image, size_t width, size_t height, 
				 	   int cx, int cy, float radius, const float4& color )	
{ 
	return cudaDrawCircle(image, width, height, imageFormatFromType<T>(), cx, cy, radius, color); 
}



/**
 * cudaDrawLine
 * @ingroup drawing
 */
cudaError_t cudaDrawLine( void* input, void* output, size_t width, size_t height, imageFormat format, 
					 int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 );
	
/**
 * cudaDrawLine
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawLine( T* input, T* output, size_t width, size_t height, 
				 	 int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 )	
{ 
	return cudaDrawLine(input, output, width, height, imageFormatFromType<T>(), x1, y1, x2, y2, color, line_width); 
}

/**
 * cudaDrawLine (in-place)
 * @ingroup drawing
 */
inline cudaError_t cudaDrawLine( void* image, size_t width, size_t height, imageFormat format, 
						   int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 )
{
	return cudaDrawLine(image, image, width, height, format, x1, y1, x2, y2, color, line_width);
}					
	
/**
 * cudaDrawLine (in-place)
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawLine( T* image, size_t width, size_t height, 
				 	 int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 )	
{ 
	return cudaDrawLine(image, width, height, imageFormatFromType<T>(), x1, y1, x2, y2, color, line_width); 
}	


/**
 * @brief 
 * 
 * @param input 
 * @param output 
 * @param width 
 * @param height 
 * @param format 
 * @param left 
 * @param top 
 * @param right 
 * @param bottom 
 * @param color 
 * @param line_color 
 * @param line_width 
 * @return cudaError_t 
 */
cudaError_t cudaDrawRect( void* input, void* output, size_t width, size_t height, imageFormat format, 
					 int left, int top, int right, int bottom, const float4& color, 
					 const float4& line_color=make_float4(0,0,0,0), float line_width=1.0f );

#endif