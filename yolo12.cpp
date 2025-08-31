#include "yolo12.h"

#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }


// const int Yolo12::num_classes;
static inline float intersection_area1(const cv::Rect_<float> &a_rect, const cv::Rect_<float> &b_rect)
{
    cv::Rect_<float> inter = a_rect & b_rect;
    return inter.area();
}

void nms_sorted_bboxes_y12(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];
            // intersection over union
            cv::Rect_<float> a_rect (a.x,a.y,a.w,a.h);
            cv::Rect_<float> b_rect (b.x,b.y,b.w,b.h);
            float inter_area = intersection_area1(a_rect, b_rect);
            float union_area = a_rect.area() + b_rect.area() - inter_area;
            float IoU = inter_area / union_area;
            if(IoU > nms_threshold) {
                keep = 0;
            }
        }

        if (keep)
            picked.emplace_back(i);
    }
}


void Yolo12::loadEngine()
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

Yolo12::Yolo12(int img_w ,int img_h)
    :img_w{img_w},img_h{img_h}
{
    engine_file_path = "/home/user/yolov12n.engine";
    loadEngine();
    buffers.reserve(engine->getNbBindings());
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];

    std::cout << "----->YOLO12 " << input_h << " "<< input_w << " "<< detection_attribute_size << " "<< num_detections << std::endl;
    // ----->YOLO12 640 640 84 8400
    num_classes = detection_attribute_size - 4;

    CUDA_CHECK(cudaMalloc(&blob, 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    cudaMallocManaged((void **) &output_buffer, detection_attribute_size * num_detections*sizeof(float));

    buffers[0] = (void *)blob;
    buffers[1] = (void *) output_buffer;

    NvBufferCreateParams input_params = {0};
    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = input_w;
    input_params.height = input_h;
    input_params.layout = NvBufferLayout_Pitch;
    input_params.colorFormat = NvBufferColorFormat_ABGR32;
    input_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT; 

    if (-1 == NvBufferCreateEx(&fd_blob, &input_params))
        std::cout<<"Failed to create NvBuffer fd_blob"<<std::endl;

    nbs = NvBufferSessionCreate();
    CUDA_CHECK(cudaStreamCreate(&stream));
}

Yolo12::~Yolo12() {

}

void Yolo12::postprocess1(std::vector<Object> &objects) {
    cudaStreamSynchronize(stream);
    cudaStreamAttachMemAsync(stream, output_buffer, 0, cudaMemAttachHost);

    std::vector<Object> proposals;
    std::vector<int> nms_result;
   
    std::vector<float> classes_scores(num_classes);

    for (int i = 0; i < num_detections; i++) {
        for (int j = 0; j < num_classes; j++)
            classes_scores[j] = output_buffer[detection_attribute_size*i + (4 + j)*num_detections];
        
        auto max_score = std::max_element(classes_scores.begin() , classes_scores.end()); 
        int argmaxVal = distance(classes_scores.begin(), max_score);

        float score = classes_scores[argmaxVal];
        
        if (score > conf_threshold) {
            const float cx = output_buffer[detection_attribute_size*i + 0*num_detections];//det_output.at<float>(0, i);
            const float cy = output_buffer[detection_attribute_size*i + 1*num_detections];//det_output.at<float>(1, i);
            const float ow = output_buffer[detection_attribute_size*i + 2*num_detections];//det_output.at<float>(2, i);
            const float oh = output_buffer[detection_attribute_size*i + 3*num_detections];//det_output.at<float>(3, i);
            Object obj;
            obj.prob = score;
            obj.label = argmaxVal;
            obj.x = static_cast<int>((cx - 0.5 * ow));
            obj.y = static_cast<int>((cy - 0.5 * oh));
            obj.w = static_cast<int>(ow);
            obj.h = static_cast<int>(oh);
            proposals.emplace_back(obj);
        }
    }

    nms_sorted_bboxes_y12(proposals, nms_result, nms_threshold);

    const float ratio_h = input_h / (float)img_h;
    const float ratio_w = input_w / (float)img_w;

    for (int i = 0; i < nms_result.size(); i++){
        Object result;
        int idx = nms_result[i];
        result = proposals[idx];
        result.id = -1;// 
        
        if (ratio_h > ratio_w) {
            result.x = result.x / ratio_w;
            result.y = (result.y - (input_h - ratio_w * img_h) / 2) / ratio_w;
            result.w = result.w / ratio_w;
            result.h = result.h / ratio_w;
        }
        else {
            result.x = (result.x - (input_w - ratio_h * img_w) / 2) / ratio_h;
            result.y = result.y / ratio_h;
            result.w = result.w / ratio_h;
            result.h = result.h / ratio_h;
        }
        objects.push_back(result);
    }
}

void Yolo12::postprocess(std::vector<Object> &objects) {
    cudaStreamSynchronize(stream);
    cudaStreamAttachMemAsync(stream, output_buffer, 0, cudaMemAttachHost);
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    std::vector<Object> proposals;
    std::vector<int> nms_result;

    const cv::Mat det_output(detection_attribute_size, num_detections, CV_32F, output_buffer);

    for (int i = 0; i < det_output.cols; ++i) {
        const cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }
    
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    int id = 0;
    for (int i = 0; i < nms_result.size(); i++){
        Object result;
        int idx = nms_result[i];
        result.id = ++id;// 
        result.label = class_ids[idx];
        result.prob = confidences[idx];

        result.x = boxes[idx].x;
        result.y = boxes[idx].y;
        result.w = boxes[idx].width;
        result.h = boxes[idx].height;


        result.x = result.x*scale + win.x;
        result.y = result.y*scale + win.y;
        result.w = result.w*scale;
        result.h = result.h*scale;
        result.is_selected = false;
        objects.push_back(result);
    }
}

std::vector<Object> Yolo12::apply(int fd) {
    std::vector<Object> objects;

    win = cv::Rect2i(0,60,1920,960);
    scale = 6;
    int src_dmabuf_fds[1];
    src_dmabuf_fds[0] = fd;

    NvBufferCompositeParams composite_params;
    memset(&composite_params, 0, sizeof(composite_params));

    composite_params.composite_flag = NVBUFFER_COMPOSITE;
    composite_params.input_buf_count = 1;
    composite_params.composite_filter[0] = NvBufferTransform_Filter_Bilinear;// NvBufferTransform_Filter_Nicest; 
    composite_params.dst_comp_rect_alpha[0] = 1.0f;

    composite_params.src_comp_rect[0].left = win.x; // vid_io_param->inConf[vid_idx].crop_left;
    composite_params.src_comp_rect[0].top = win.y;
    composite_params.src_comp_rect[0].width = win.width;
    composite_params.src_comp_rect[0].height = win.height;

    composite_params.dst_comp_rect[0].top = 0; 
    composite_params.dst_comp_rect[0].left = 0; 
    composite_params.dst_comp_rect[0].width = input_w;
    composite_params.dst_comp_rect[0].height = input_h; 
    composite_params.session = nbs;
    NvBufferComposite(src_dmabuf_fds, fd_blob, &composite_params);

    CudaProcess cup{fd_blob};
    auto blob_ptr = cup.getImgPtr();
    
    cudaBlobFromImageRGB(blob_ptr, blob, cup.getPitch()/4);
    cup.freeImage();

    context->enqueueV2(buffers.data(), stream, nullptr);
    postprocess(objects);

    return objects;
}
