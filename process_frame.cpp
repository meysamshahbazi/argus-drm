#include "process_frame.h"


ProcessFrame::ProcessFrame(){
    auto win = cv::Rect2i(0,28,1920,1024); // 128*128
    car_detect = new Yolo12(1920,1080,"/home/user/best.engine", win);
    
    auto win_plate = cv::Rect2i(0,60,1920,960);// cv::Rect2i(0,0,160,160); // 320*320
    plate_detect = new Yolo12(160,160,"/home/user/plate.engine", win_plate);

    new_car_det = false;
    quit = false; 

    NvBufferCreateParams cParams = {0};
    cParams.colorFormat = NvBufferColorFormat_ABGR32; // because the histogram qualizer works with this!
    cParams.width = 160;
    cParams.height = 160;
    cParams.layout = NvBufferLayout_Pitch;
    cParams.payloadType = NvBufferPayload_SurfArray;
    cParams.nvbuf_tag = NvBufferTag_VIDEO_DEC;

    if (-1 == NvBufferCreateEx(&car_fd, &cParams) ){
        std::cout<<"Failed to create buffers "<<std::endl;
    }

    memset(&transParams, 0, sizeof(transParams));
    transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    transParams.transform_filter = NvBufferTransform_Filter_Nicest;
    transParams.session = nbs;

    lpr = new LPR();
    run();
}

ProcessFrame::~ProcessFrame(){
      
}

void ProcessFrame::run() {
    pthread_create(&ptid_run, NULL, (THREADFUNCPTR)&thread_func, (void *)this);
}

void* ProcessFrame::thread_func (void* arg) {
    pthread_detach(pthread_self());
    ProcessFrame* thiz = (ProcessFrame*) arg;
    thiz->func_plate();
    pthread_exit(NULL);
}

bool ProcessFrame::func_plate() {
    // while (!quit) {
    //     waitForNewDet();
    //     if (car_objs.size() == 0)
    //         continue;

    //     auto win_ = car_objs[0].getRect();
    //     auto plate_objs = plate_detect->apply(argb_fd,win_);
        

    //     for (auto obj : plate_objs)
    //         std::cout << "**** plate_objs " << obj.getRect() << std::endl;

    // }
}

void ProcessFrame::waitForNewDet() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    t.tv_sec += 0;
    t.tv_nsec += 40000000;
    
    if (!new_car_det)
        pthread_cond_timedwait(&new_car_det_cond, 
            &new_car_det_mutex, &t);

    new_car_det = false;
}


int ProcessFrame::apply(int fd) {
    auto start = std::chrono::system_clock::now();
    car_objs = car_detect->apply(fd); 
    auto end = std::chrono::system_clock::now();
    auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "detect: " << micro/1000.0f << std::endl;
    std::vector<Object> plate_objs;

    if (car_objs.size() > 0) {
        argb_fd = fd;
        auto win_ = car_objs[0].getRect();

        plate_objs = plate_detect->apply(argb_fd,win_);
        // std::cout << "plate_objs size: " <<  plate_objs.size() << std::endl;
        for (auto obj : plate_objs){
            auto start = std::chrono::system_clock::now();
            cv::Rect2i win2 =  cv::Rect2i(obj.x - 6, obj.y - 6, obj.w+12, obj.h+12);
            lpr->apply(argb_fd, win2);
            auto end = std::chrono::system_clock::now();
            auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            std::cout << "lpr: " << micro/1000.0f << std::endl;
        }        
        // pthread_mutex_lock(&new_car_det_mutex);
        // new_car_det = true; 
        // pthread_cond_broadcast(&new_car_det_cond);
        // pthread_mutex_unlock(&new_car_det_mutex);
    }

    
    CudaProcess cup(fd);
    auto img_ptr = cup.getImgPtr();
    for (auto obj : car_objs){
        // std::cout << obj.label << ", " << obj.x<< ", " << obj.y << ", " << obj.x + obj.w  << ", " <<  obj.y + obj.h << "\n";
        cudaDrawRect(img_ptr, img_ptr , 1920, 1080, IMAGE_RGBA8, obj.x, obj.y, obj.x + obj.w , obj.y + obj.h, 
            make_float4(0.0, 0.0, 0.0, 0.0), make_float4(200.0f, 0.0f, 0.0f, 255.0f), 1 );
    }
    for (auto obj : plate_objs){
        // std::cout << "w2 is" << obj.getRect() <<std::endl;
        // std::cout << obj.label << ", " << obj.x<< ", " << obj.y << ", " << obj.x + obj.w  << ", " <<  obj.y + obj.h << "\n";
        cudaDrawRect(img_ptr, img_ptr , 1920, 1080, IMAGE_RGBA8, obj.x - 6, obj.y -6, obj.x + obj.w+6 , obj.y + obj.h+6, 
            make_float4(0.0, 0.0, 0.0, 0.0), make_float4(0.0f, 0.0f, 255.0f, 255.0f), 1 );
    }
    // std::cout << std::endl;
    cup.freeImage();
    
    return fd;
}