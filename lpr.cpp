
#include "lpr.h"
#include <bits/stdc++.h>

LPR::LPR() {
    character = {"[CTCblank]", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "s", "t", "u", "v", "x", "y", "z"};
    engine_file_path = "/home/user/best_accuracy.engine";
    loadEngine();
    std::cout << engine->getNbBindings() << std::endl;
    // printDim(d1);
    d1 = engine->getBindingDimensions(1).d[1];
    d2 = engine->getBindingDimensions(1).d[2];
    // printDim(d2);
    cudaStreamCreate(&stream);
    cudaMalloc(&blob, 1 * 32 * 100 * sizeof(float));
    // Initialize output buffer
    cudaMallocManaged((void **) &output_buffer, d1 * d2*sizeof(float));
    buffers.reserve(engine->getNbBindings());
    buffers[0] = (void *)blob;
    buffers[1] = (void *) output_buffer;

    NvBufferCreateParams input_params = {0};
    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = 100;
    input_params.height = 32;
    input_params.layout = NvBufferLayout_Pitch;
    input_params.colorFormat = NvBufferColorFormat_ARGB32;
    input_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT; 

    if (-1 == NvBufferCreateEx(&fd_blob, &input_params))
        std::cout<<"Failed to create NvBuffer fd_blob"<<std::endl;

    nbs = NvBufferSessionCreate();
    

    // for (int i = 0; i < 1000; i ++) {
    //     auto start = std::chrono::system_clock::now();
    //     context->enqueueV2(buffers.data(), stream, nullptr);
    //     cudaStreamSynchronize(stream);
    //     auto end = std::chrono::system_clock::now();
    //     auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //     std::cout << "lpr: " << micro/1000.0f << std::endl;
    // }
}

LPR::~LPR() {
    
}

std::vector<std::string> LPR::apply(int fd, cv::Rect2i &win_) {
    win = win_;
    return apply(fd);
}

std::vector<std::string> LPR::apply(int fd) {
    std::vector<std::string> result;

    int src_dmabuf_fds[1];
    src_dmabuf_fds[0] = fd;

    NvBufferCompositeParams composite_params;
    memset(&composite_params, 0, sizeof(composite_params));

    composite_params.composite_flag = NVBUFFER_COMPOSITE;
    composite_params.input_buf_count = 1;
    composite_params.composite_filter[0] = NvBufferTransform_Filter_Bilinear;// NvBufferTransform_Filter_Nicest; 
    composite_params.dst_comp_rect_alpha[0] = 1.0f;
    composite_params.src_comp_rect[0].left = win.x; 
    composite_params.src_comp_rect[0].top = win.y;
    composite_params.src_comp_rect[0].width = win.width;
    composite_params.src_comp_rect[0].height = win.height;

    composite_params.dst_comp_rect[0].top = 0; 
    composite_params.dst_comp_rect[0].left = 0; 
    composite_params.dst_comp_rect[0].width = 100;
    composite_params.dst_comp_rect[0].height = 32; 
    composite_params.session = nbs;
    NvBufferComposite(src_dmabuf_fds, fd_blob, &composite_params);

    CudaProcess cup{fd_blob};
    auto blob_ptr = cup.getImgPtr();
    cudaBlobFromImageGray(blob_ptr, blob, cup.getPitch()/4);
    cup.freeImage();

    context->enqueueV2(buffers.data(), stream, nullptr);
    postprocess(result);

    return result;
}

void LPR::postprocess(std::vector<std::string> &result) {
    std::vector<int> out_indexs;
    cudaStreamSynchronize(stream);
    cudaStreamAttachMemAsync(stream, output_buffer, 0, cudaMemAttachHost);
    for (int i = 0; i < d1; i++) {
        std::vector<float> row;
        for (int j = 0; j<d2; j++)
            row.push_back(output_buffer[i*d2+j]);

        auto max_itr = std::max_element(row.begin(), row.end());
        
        int max_index = std::distance(row.begin(), max_itr);
        out_indexs.push_back(max_index);
        // if (max_index != 0)
        //     std::cout << character[max_index] << " , ";
    }

    std::vector<int> out_cleaned; 
    for (int i = 0; i < d1; i++) {
        if (out_indexs[i] != 0) {
            if (i == 0)
                out_cleaned.push_back(out_indexs[i]);
            else {
                if (out_indexs[i] != out_indexs[i-1] )
                    out_cleaned.push_back(out_indexs[i]);
            }
        }
    }

    for (auto i : out_cleaned)
        std::cout << character[i] << " , ";
    std::cout << std::endl;
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