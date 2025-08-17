#include "cuproc.h"

CudaProcess::CudaProcess() :fd{-1}
{
    
}

CudaProcess::CudaProcess(int fd) :fd{fd}
{

}

CudaProcess::~CudaProcess()
{

}

void CudaProcess::setFd(int fd)
{
    this->fd = fd;
}

void* CudaProcess::getImgPtr()
{
    image = NvEGLImageFromFd(NULL, fd);
    cudaFree(0);
    status = cuGraphicsEGLRegisterImage(&pResource, image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS) {
        printf("cuGraphicsEGLRegisterImage failed in : %d, cuda process stop\n",status);
        return NULL;
    }
    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
    if (status != CUDA_SUCCESS) {
        printf("cuGraphicsSubResourceGetMappedArray failed\n");
        return NULL;
    }
    return eglFrame.frame.pPitch[0];
}

CuImg CudaProcess::getCuImg() {
    image = NvEGLImageFromFd(NULL, fd);
    cudaFree(0);
    status = cuGraphicsEGLRegisterImage(&pResource, image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS) {
        printf("cuGraphicsEGLRegisterImage failed in : %d, cuda process stop\n",status);
        return cu_img;
    }
    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
    if (status != CUDA_SUCCESS) {
        printf("cuGraphicsSubResourceGetMappedArray failed\n");
        return cu_img;
    }
    cu_img.img_ptr = eglFrame.frame.pPitch[0];
    cu_img.width = eglFrame.width;
    cu_img.height = eglFrame.height;
    cu_img.pitch = eglFrame.pitch;
    return cu_img;
}

void CudaProcess::freeImage()
{
    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS) {
        printf("failed after memcpy\n");
        exit(-1);
    }

    status = cuGraphicsUnregisterResource(pResource);
    if (status != CUDA_SUCCESS) {
        printf("cuGraphicsEGLUnRegisterResource failed: %d\n", status);
    }
    NvDestroyEGLImage(NULL, image);
}

void CudaProcess::printImageInfo() const
{
    printf("-------------------------------------------------\n");
    printf("width: %d\t",eglFrame.width);
    printf("height: %d\t",eglFrame.height);
    printf("depth: %d\t",eglFrame.depth);
    printf("pitch: %d\t",eglFrame.pitch);
    printf("planeCount: %d\t",eglFrame.planeCount);
    printf("numChannels: %d\t",eglFrame.numChannels);
    printf("eglColorFormat: %d\t",eglFrame.eglColorFormat);
    printf("cuFormat: %d\t",eglFrame.cuFormat);
    printf("\n-------------------------------------------------\n");
}

int CudaProcess::getPitch()
{
    return eglFrame.pitch;
}

int CudaProcess::getWidth() {
    return eglFrame.width;
}

int CudaProcess::getHeight() {
    return eglFrame.height;
}

