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

#include <opencv2/dnn.hpp>


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


struct Object
{
    // cv::Rect_<float> rect;
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
    int id;

    bool is_selected{false};

    Object() {

    }

    static Object createObject(const cv::Rect &rect, int id=-1, int label= -1, float prob = 0) {
        Object obj;
        obj.x = rect.x;
        obj.y = rect.y;
        obj.w = rect.width;
        obj.h = rect.height;
        obj.id = id;
        obj.label = label;
        obj.prob = prob;
        return obj;
    }

    cv::Point2i tl() {
        return {static_cast<int>(x), static_cast<int>(y)};
    }

    cv::Point2i br() {
        return {static_cast<int>(x + w), static_cast<int>(y + h)};
    }

    cv::Point2i getCenter() {
        return {static_cast<int>(x + w/2), static_cast<int>(y + h/2)};
    }
    cv::Rect2i getRect() {
        return cv::Rect2i(x,y,w,h);
    }
};


class Yolo12 {
private:
    // Model parameters
    int input_w;  // Input image width expected by the model
    int input_h;  // Input image height expected by the model
    int num_detections;  // Number of detections output by the model
    int detection_attribute_size;  // Attributes (e.g., bbox, class) per detection
    int num_classes = 80;  // Number of classes (e.g., COCO dataset has 80 classes)
    static const int nc{10}; // number of class

    // Maximum supported image size (used for memory allocation checks)

    // Confidence threshold for filtering detections
    float conf_threshold = 0.3f; //0.3

    // Non-Maximum Suppression (NMS) threshold to remove duplicate boxes
    float nms_threshold = 0.4f; // 0.4


    int img_w;
    int img_h;
    void* iresized{nullptr};
    cudaStream_t stream;
    std::string engine_file_path;
    Logger gLogger;
    std::unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context;
    std::unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine;
    std::vector<void*> buffers;
    float* output_buffer;
    float* blob;
    
    cv::Rect2i win;
    float scale_x;
    float scale_y;

    int fd_blob {-1};
    NvBufferSession nbs;
    
    void loadEngine();
    void cudaBlobFromImageRGB(void* img_rgb, float* blob, int pitch);
    void cudaCropAndBlobFromImageRGB(void* img_rgb, float* blob);
    void postprocess1(std::vector<Object> &objects); // orginal version
    void postprocess(std::vector<Object> &objects);

public:
    ~Yolo12();
    Yolo12(int img_w, int img_h, std::string engine_file_path, cv::Rect2i &win);
    std::vector<Object> apply(int fd);

};





