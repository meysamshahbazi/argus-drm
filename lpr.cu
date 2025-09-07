
#include "lpr.h"
#include "cudaUtility.h"

__global__ void gpuFillBlobFromGray(uchar4* im_gray, int blob_width, int blob_height, int pitch,float* blob)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > blob_width || y > blob_height )
        return;

    float gray_val = static_cast<float>( im_gray[ y*pitch+x ].x -127)/255.0;// R
    gray_val += static_cast<float>( im_gray[ y*pitch+x ].y -127)/255.0;// G
    gray_val += static_cast<float>( im_gray[ y*pitch+x ].z -127)/255.0;// G
    blob[ y*blob_width + x ] = gray_val/3.0f;// R
 
}


void LPR::cudaBlobFromImageGray(void* img_gray, float* blob, int pitch)
{
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(100,blockDim.x), iDivUp(32,blockDim.y));

    gpuFillBlobFromGray<<<gridDim, blockDim, 0, stream>>>((uchar4*) img_gray, 100, 32, pitch, blob);
    cudaDeviceSynchronize();
    cudaError_t res;
    res = cudaGetLastError();
    if(res != cudaSuccess ) 
        printf("error in cuda gpuFillBlobFromRGB %d \n",res);
}
