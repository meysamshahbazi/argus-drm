#ifndef _CUDA_DEINTERLACE_H_
#define _CUDA_DEINTERLACE_H_
#include "cudaUtility.h"

cudaError_t cudaDeinterlace(void* input_cur,void* filed_buf, void* output,
unsigned char *bottom_field, size_t width, size_t height, cudaStream_t stream);

#endif
