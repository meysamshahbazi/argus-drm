#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "argus_capture.h"
#include "video_encoder.h"
#include "process_frame.h"
#include "udp_client.h"

class PlateReader {
private:
    VideoEncoder* videoencoder{nullptr};
    ArgusCapture *ac{nullptr};
    GstRtp *gst_rtp{nullptr};
    UdpClient *udp{nullptr};
    ProcessFrame *pf{nullptr};

    int argb_fd;
    NvBufferTransformParams transParams;
    
    pthread_t ptid_enc, ptid_proc;
    static void* func_enc(void* arg);
    void run_enc_loop();

    static void* func_proc(void* arg);
    void run_proc_loop();

    std::chrono::_V2::system_clock::time_point t_perv;

public:
    PlateReader();
    ~PlateReader();
    void run();
};

