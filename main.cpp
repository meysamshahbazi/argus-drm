#include <iostream>
#include <memory>
#include <string>

#include <gst/gst.h>

#include "NvDrmRenderer.h"
#include "tegra_drm_nvdc.h"

#include "argus_capture.h"
#include "cuproc.h"
#include "cudaDraw.h"

#include "yolo12.h"
#include "lpr.h"

int main(int argc, char** argv) {
    // saveEngineFile("/home/user/best.onnx","/home/user/best.engine");
    // saveEngineFile("/home/user/best_accuracy.onnx","/home/user/best_accuracy.engine");
    // return -1;

    // LPR lpr;    
    auto win = cv::Rect2i(0,60,1920,960); // 320*320
    Yolo12 car_detect(1920,1080,"/home/user/best.engine", win);

    auto win_plate = cv::Rect2i(0,0,160,960); // 320*320

    Yolo12 plate_detect(160,160,"/home/user/plate.engine", win_plate);

    NvBufferSession nbs;
    nbs = NvBufferSessionCreate();

    NvBufferTransformParams transParams;
    memset(&transParams, 0, sizeof(transParams));
    transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    transParams.transform_filter = NvBufferTransform_Filter_Nicest;
    transParams.session = nbs;


    static const int NUM_RENDER_BUFFERS{3};

    struct drm_tegra_hdr_metadata_smpte_2086 drm_metadata;
    NvDrmRenderer *hdmi = NvDrmRenderer::createDrmRenderer("renderer0", 1920, 1080, 0, 0,
            /*connector*/ 0, /*crtc*/ 0, /*plane_id*/ 0, drm_metadata, true);
    hdmi->setFPS(30);

    NvBufferCreateParams cParams = {0};
    cParams.colorFormat = NvBufferColorFormat_ABGR32; // because the histogram qualizer works with this!
    cParams.width = 1920;
    cParams.height = 1080;
    cParams.layout = NvBufferLayout_Pitch;
    cParams.payloadType = NvBufferPayload_SurfArray;
    cParams.nvbuf_tag = NvBufferTag_VIDEO_DEC;


    int render_fd_arr[NUM_RENDER_BUFFERS];
    /* Create pitch linear buffers for renderring */
    for (int index = 0; index < NUM_RENDER_BUFFERS; index++) {
        if (-1 == NvBufferCreateEx(&render_fd_arr[index], &cParams) ){
            std::cout<<"Failed to create buffers "<<std::endl;
            return -1;
        }
    }

    int argb_fd;
    if (-1 == NvBufferCreateEx(&argb_fd, &cParams) ){
        std::cout<<"Failed to create buffers "<<std::endl;
        return -1;
    }

    ArgusCapture ac;
    ac.run();

    int render_cnt = 0;
    int render_fd;

    while(1) {
        

        int fd_ = ac.getFd();
        if (fd_ == -1) continue;

        NvBufferTransform(fd_, argb_fd , &transParams);

        auto start = std::chrono::system_clock::now();
        auto objs = car_detect.apply(argb_fd);
        auto end = std::chrono::system_clock::now();

        // for (auto obj : objs){
        //     std::cout << obj.label << ", ";
        // }
        // std::cout << std::endl;

        auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "detect: " << micro/1000.0f << std::endl;
        
        CudaProcess cup(argb_fd);
        auto img_ptr = cup.getImgPtr();

        for (auto obj : objs){
            std::cout << obj.label << ", " << obj.x<< ", " << obj.y << ", " << obj.x + obj.w  << ", " <<  obj.y + obj.h << " ";
            cudaDrawRect(img_ptr, img_ptr , 1920, 1080, IMAGE_RGBA8, obj.x, obj.y, obj.x + obj.w , obj.y + obj.h, 
                make_float4(0.0, 0.0, 0.0, 0.0), make_float4(200.0f, 0.0f, 0.0f, 255.0f), 1 );
        }
        cup.freeImage();
       

        if (render_cnt < NUM_RENDER_BUFFERS ) {
            render_fd = render_fd_arr[render_cnt];
            render_cnt++;
        } 
        else {
            render_fd = hdmi->dequeBuffer();
        }
        
        NvBufferTransform(argb_fd, render_fd , &transParams);    
        hdmi->enqueBuffer(render_fd);
    }

    return 0;
}
