#include "process_frame.h"


ProcessFrame::ProcessFrame(){
    auto win = cv::Rect2i(0,60,1920,960); // 320*320
    car_detect = new Yolo12(1920,1080,"/home/user/best.engine", win);
    
    auto win_plate = cv::Rect2i(0,0,160,960); // 320*320
    plate_detect = new Yolo12(160,160,"/home/user/plate.engine", win_plate);
}

ProcessFrame::~ProcessFrame(){
      
}

int ProcessFrame::apply(int fd) {
    auto start = std::chrono::system_clock::now();
    auto objs = car_detect->apply(fd);
    auto end = std::chrono::system_clock::now();

    // for (auto obj : objs){
    //     std::cout << obj.label << ", ";
    // }
    // std::cout << std::endl;

    auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "detect: " << micro/1000.0f << std::endl;
    
    CudaProcess cup(fd);
    auto img_ptr = cup.getImgPtr();

    for (auto obj : objs){
        std::cout << obj.label << ", " << obj.x<< ", " << obj.y << ", " << obj.x + obj.w  << ", " <<  obj.y + obj.h << " ";
        cudaDrawRect(img_ptr, img_ptr , 1920, 1080, IMAGE_RGBA8, obj.x, obj.y, obj.x + obj.w , obj.y + obj.h, 
            make_float4(0.0, 0.0, 0.0, 0.0), make_float4(200.0f, 0.0f, 0.0f, 255.0f), 1 );
    }
    cup.freeImage();
    
    return fd;
}