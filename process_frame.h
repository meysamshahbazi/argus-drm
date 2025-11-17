#pragma once 

#include "cuproc.h"
#include "cudaDraw.h"

#include "yolo12.h"
#include "lpr.h"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "sysio.h"

#include "udp_client.h"


class ProcessFrame {
private:
    Yolo12 *car_detect{nullptr};
    Yolo12 *plate_detect{nullptr};
    LPR *lpr;
    bool quit{false};
    pthread_t ptid_run;
    pthread_cond_t new_car_det_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t new_car_det_mutex = PTHREAD_MUTEX_INITIALIZER;
    bool new_car_det{false};

    std::vector<Object> car_objs;
    NvBufferSession nbs;
    NvBufferTransformParams transParams;

    int car_fd;
    int argb_fd;

    void waitForNewDet();

    void run();
    bool func_plate();
    static void* thread_func(void* arg);

public:
    ProcessFrame();
    ~ProcessFrame();
    PlateResult apply(int fd);

};