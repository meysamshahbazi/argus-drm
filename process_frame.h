#pragma once 

#include "cuproc.h"
#include "cudaDraw.h"

#include "yolo12.h"
#include "lpr.h"

class ProcessFrame {
private:
    Yolo12 *car_detect{nullptr};
    Yolo12 *plate_detect{nullptr};
    LPR *lpr;

public:
    ProcessFrame();
    ~ProcessFrame();
    int apply(int fd);
};