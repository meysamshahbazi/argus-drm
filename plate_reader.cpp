#include "plate_reader.h"

void encode_callback(int i, void* arg)
{

}


PlateReader::PlateReader() {
    videoencoder = new VideoEncoder("enc0", 1920, 1080, V4L2_PIX_FMT_H264);
    videoencoder->setBufferDoneCallback(&encode_callback, nullptr);
    bool res = videoencoder->initialize();

    ac = new ArgusCapture();
    ac->run();

    pf = new ProcessFrame();
    udp = new UdpClient();

    gst_rtp = videoencoder->getRtp();

    NvBufferSession nbs;
    nbs = NvBufferSessionCreate();

    
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

    if (-1 == NvBufferCreateEx(&argb_fd, &cParams) ) {
        std::cout << "Failed to create buffers argb_fd" << std::endl;
    }

}

PlateReader::~PlateReader() {
    delete ac;
    delete pf;
    delete udp;
}


void* PlateReader::func_enc(void* arg) {
    pthread_detach(pthread_self());
    PlateReader* thiz = (PlateReader*) arg;
    thiz->run_enc_loop();
    pthread_exit(NULL);
}

void PlateReader::run_enc_loop() {
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
}


void* PlateReader::func_proc(void* arg) {
    pthread_detach(pthread_self());
    PlateReader* thiz = (PlateReader*) arg;
    thiz->run_proc_loop();
    pthread_exit(NULL);
}

void PlateReader::run_proc_loop() {
    uint32_t last_fc =0;
    while(1) {
        struct timespec t;
        clock_gettime(CLOCK_REALTIME, &t);
        t.tv_sec += 0;
        t.tv_nsec += 40000000;
        
        pthread_cond_timedwait(&ac->new_frame_cond, 
                &ac->new_frame_mutex, &t);

        int fd_ = ac->getFd();
        auto fc = ac->getFrameCnt();
        if (fc == last_fc){
            usleep(2000);
            continue;
        }
        last_fc = fc;

        if (fd_ == -1) continue;
        NvBufferTransform(fd_, argb_fd , &transParams);
        auto md = pf->apply(argb_fd);
        md.frame_cnt = fc;
        udp->sendMetaData(md);     
    }
}


void PlateReader::run() {
    pthread_create(&ptid_enc, NULL, (THREADFUNCPTR)&func_enc, this);
    pthread_create(&ptid_proc, NULL, (THREADFUNCPTR)&func_proc, this);
    pthread_join(ptid_enc, NULL);
}