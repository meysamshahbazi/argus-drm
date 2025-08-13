#ifndef _CUPROC_H_
#define _CUPROC_H_

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "nvbuf_utils.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cudaEGL.h"

struct CuImg
{
    void * img_ptr;
    int width,height,pitch;
};

class CudaProcess
{
public:
    CudaProcess();
    CudaProcess(int fd);
    ~CudaProcess();
    void* getImgPtr();
    void freeImage();
    void setFd(int fd);
    int getPitch();
    int getWidth();
    int getHeight();
    void printImageInfo() const;
    CuImg getCuImg();
private:
    int fd;
    EGLImageKHR image;
    CUresult status;
    CUeglFrame eglFrame;
    CUgraphicsResource pResource = NULL;
    CuImg cu_img;
};

#endif