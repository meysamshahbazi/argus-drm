
#include "yolo12.h"
#include "cudaUtility.h"

__global__ void gpuFillBlobFromRGB_yolo12(uchar4* im_rgba, int blob_width, int blob_height, int pitch,float* blob)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    blob[ y*blob_width + x + 0*blob_width*blob_height] = static_cast<float>(im_rgba[y*pitch+x].x)/255.0f;// R
    blob[ y*blob_width + x + 1*blob_width*blob_height] = static_cast<float>(im_rgba[y*pitch+x].y)/255.0f;// G
    blob[ y*blob_width + x + 2*blob_width*blob_height] = static_cast<float>(im_rgba[y*pitch+x].z)/255.0f;// B
}


void Yolo12::cudaBlobFromImageRGB(void* img_rgb, float* blob, int pitch)
{
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(input_w,blockDim.x), iDivUp(input_h,blockDim.y));

    gpuFillBlobFromRGB_yolo12<<<gridDim, blockDim, 0, stream>>>((uchar4*) img_rgb, input_w, input_h, pitch, blob);
    cudaDeviceSynchronize();
    cudaError_t res;
    res = cudaGetLastError();
    if(res != cudaSuccess ) 
        printf("error in cuda gpuFillBlobFromRGB %d \n",res);
}


__global__ void CropAndRGBAToRGBGPU_yolo12(uchar4* srcImage, float* dstImage, int width, int height, 
	int x0, int y0, int w, int h)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel_src = (y+y0) * width + (x+x0);

	if( x >= width || y >= height )
		return; 

	const uchar4 px = srcImage[pixel_src];

    dstImage[ y*w + x + 0*w*h] = static_cast<float>(px.x)/255.0f; // R    
    dstImage[ y*w + x + 1*w*h] = static_cast<float>(px.y)/255.0f; // G
    dstImage[ y*w + x + 2*w*h] = static_cast<float>(px.z)/255.0f; // B
}


void Yolo12::cudaCropAndBlobFromImageRGB(void* img_rgb, float* blob) {
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(input_w,blockDim.x), iDivUp(input_h,blockDim.y));

    CropAndRGBAToRGBGPU_yolo12<<<gridDim, blockDim, 0, stream>>>((uchar4*) img_rgb, blob, img_w, img_h, win.x, win.y, win.width, win.height);
    cudaDeviceSynchronize();
    cudaError_t res;
    res = cudaGetLastError();
    if(res != cudaSuccess ) 
        printf("error in cuda cudaCropAndBlobFromImageRGB %d \n",res);
}

