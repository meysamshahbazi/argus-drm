#include <iostream>
#include <memory>
#include <string>

#include <gst/gst.h>

#include "NvDrmRenderer.h"
#include "tegra_drm_nvdc.h"

#include "argus_capture.h"
#include "video_encoder.h"
#include "process_frame.h"
#include "udp_client.h"


VideoEncoder* videoencoder;
ArgusCapture *ac;
GstRtp *gst_rtp;
UdpClient *udp;
ProcessFrame *pf;

std::chrono::_V2::system_clock::time_point t_perv;



void encode_callback(int i, void* arg)
{

}

bool run() {
    
}

void* func_grab_run(void* arg) {
    pthread_detach(pthread_self());
    gst_rtp = videoencoder->getRtp();
    uint32_t last_fc{0};
    while (1) {
        struct timespec t;
        clock_gettime(CLOCK_REALTIME, &t);
        t.tv_sec += 0;
        t.tv_nsec += 40000000;
        
        if (!ac->new_frame_flag)
            pthread_cond_timedwait(&ac->new_frame_cond, 
                &ac->new_frame_mutex, &t);

        ac->new_frame_flag = false;

        int fd_ = ac->getFd();
        if (fd_ == -1) continue;
        uint32_t fc = ac->getFrameCnt();

        if (fc == last_fc)
            continue;
        
        last_fc = fc;
        videoencoder->encodeFromFd(fd_);
        gst_rtp->setFrameCnt(fc);

        // auto t_now = std::chrono::system_clock::now();
        // auto micro = std::chrono::duration_cast<std::chrono::microseconds>(t_now -t_perv).count();
        // t_perv = t_now;
        // std::cout << "du: " << micro/1000.0f << std::endl;
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

    NvBufferCreateParams cParams = {0};
    cParams.colorFormat = NvBufferColorFormat_ABGR32; // because the histogram qualizer works with this!
    cParams.width = 1920;
    cParams.height = 1080;
    cParams.layout = NvBufferLayout_Pitch;
    cParams.payloadType = NvBufferPayload_SurfArray;
    cParams.nvbuf_tag = NvBufferTag_VIDEO_DEC;

    int argb_fd;
    if (-1 == NvBufferCreateEx(&argb_fd, &cParams) ){
        std::cout<<"Failed to create buffers "<<std::endl;
        return -1;
    }

    ac = new ArgusCapture();
    ac->run();

    pf = new ProcessFrame();
    udp = new UdpClient();

    pthread_create(&ptid_run, NULL, (THREADFUNCPTR)&func_grab_run, nullptr);
    uint32_t last_fc2 =0;
    while(1) {
        int fd_ = ac->getFd();
        auto fc = ac->getFrameCnt();
        if (fc == last_fc2){
            usleep(2000);
            continue;
        }
        last_fc2 = fc;

        if (fd_ == -1) continue;
        NvBufferTransform(fd_, argb_fd , &transParams);
        auto md = pf->apply(argb_fd);
        md.frame_cnt = fc;
        udp->sendMetaData(md);     
    }

    return 0;
}
