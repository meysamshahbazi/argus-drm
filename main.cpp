#include <iostream>
#include <memory>
#include <string>

#include <gst/gst.h>

#include "NvDrmRenderer.h"
#include "tegra_drm_nvdc.h"

#include "argus_capture.h"
#include "video_encoder.h"
#include "process_frame.h"


VideoEncoder* videoencoder;
ArgusCapture *ac;

void encode_callback(int i, void* arg)
{

}

bool run() {
    
}

void* func_grab_run(void* arg) {
    pthread_detach(pthread_self());
    
    while (1) {
        int fd_ = ac->getFd();
        if (fd_ == -1) continue;
        // NvBufferTransform(fd_, argb_fd , &transParams);
        // argb_fd = pf.apply(argb_fd);
        videoencoder->encodeFromFd(fd_);
    }


    pthread_exit(NULL);
}



int main(int argc, char** argv) {
    // saveEngineFile("/home/user/best.onnx","/home/user/best.engine");
    // saveEngineFile("/home/user/best_accuracy.onnx","/home/user/best_accuracy.engine");
    // return -1;

    gst_init(&argc, &argv);

    pthread_t ptid_run;

    videoencoder = new VideoEncoder("enc0", 1920, 1080, V4L2_PIX_FMT_H264);
    videoencoder->setBufferDoneCallback(&encode_callback, nullptr);
    bool res = videoencoder->initialize();

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

    ac = new ArgusCapture();
    ac->run();

    int render_cnt = 0;
    int render_fd;

    ProcessFrame pf;


    pthread_create(&ptid_run, NULL, (THREADFUNCPTR)&func_grab_run, nullptr);

    while(1) {
   
        // if (render_cnt < NUM_RENDER_BUFFERS ) {
        //     render_fd = render_fd_arr[render_cnt];
        //     render_cnt++;
        // } 
        // else {
        //     render_fd = hdmi->dequeBuffer();
        // }
        
        int fd_ = ac->getFd();
        if (fd_ == -1) continue;
        NvBufferTransform(fd_, argb_fd , &transParams);
        argb_fd = pf.apply(argb_fd);
        // videoencoder->encodeFromFd(fd_);
        // usleep(30000);
        // NvBufferTransform(argb_fd, render_fd , &transParams);    
        // NvBufferTransform(fd_, render_fd , &transParams);    
        // hdmi->enqueBuffer(argb_fd);
        
    }

    return 0;
}
