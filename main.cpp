#include <iostream>
#include <memory>
#include <string>

#include <gst/gst.h>
#include "plate_reader.h"

int main(int argc, char** argv) {

    // saveEngineFile("/home/user/best.onnx","/home/user/best.engine");
    // saveEngineFile("/home/user/best_accuracy.onnx","/home/user/best_accuracy.engine");
    // return -1;

    gst_init(&argc, &argv);

    PlateReader pr;
    pr.run();

    return 0;
}
