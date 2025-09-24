#pragma once 

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "logging.h"
#include "cuproc.h"
#include "trt_utils.h"
#include <numeric>
#include <chrono>
#include "nvbuf_utils.h"
#include "cudaResize.h"


class LPR{
private:
    void loadEngine();

    std::string engine_file_path;
    cudaStream_t stream;
    Logger gLogger;
    std::unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context;
    std::unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine;
    std::vector<void*> buffers;
    float* output_buffer;
    float* blob;

    int fd_blob {-1};
    NvBufferSession nbs;

    cv::Rect2i win;

    std::vector<std::string> character;
    void postprocess(std::vector<std::string> &objects);
    void cudaBlobFromImageGray(void* img_gray, float* blob, int pitch);

    int d1,d2; // dimention of output buffer
public:
    LPR();
    ~LPR();
    std::vector<std::string> apply(int fd);
    std::vector<std::string> apply(int fd, cv::Rect2i &win);

};

