
#include "lpr.h"

LPR::LPR() {
    engine_file_path = "/home/user/best_accuracy.engine";
    loadEngine();
    std::cout << engine->getNbBindings() << std::endl;
    auto d1 = engine->getBindingDimensions(0);
    printDim(d1);
    auto d2 = engine->getBindingDimensions(1);
    printDim(d2);
    cudaStreamCreate(&stream);
    cudaMalloc(&blob, 1 * 32 * 100 * sizeof(float));
    // Initialize output buffer
    cudaMallocManaged((void **) &output_buffer, 26 * 32*sizeof(float));
    buffers.reserve(engine->getNbBindings());
    buffers[0] = (void *)blob;
    buffers[1] = (void *) output_buffer;

    for (int i = 0; i < 1000; i ++) {
        auto start = std::chrono::system_clock::now();
        context->enqueueV2(buffers.data(), stream, nullptr);
        cudaStreamSynchronize(stream);
        auto end = std::chrono::system_clock::now();
        auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "lpr: " << micro/1000.0f << std::endl;
    }
}

LPR::~LPR() {
    
}


void LPR::loadEngine()
{
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime{nvinfer1::createInferRuntime(gLogger)};
    assert(runtime != nullptr);
    engine.reset(runtime->deserializeCudaEngine(trtModelStream, size));
    assert(engine != nullptr); 
    context.reset( engine->createExecutionContext() );
    assert(context != nullptr);
    delete[] trtModelStream;
}