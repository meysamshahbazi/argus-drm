
#include <string.h>
#include <assert.h>
#include "video_encoder.h"
#include <linux/videodev2.h>
#include "sysio.h"

#define CHECK_ERROR(expr) \
    do { \
        if ((expr) < 0) { \
            m_VideoEncoder->abort(); \
            printf(#expr " failed"); \
        } \
    } while (0);

// WAR: Since dqBuffer only happens when a new qBuffer is required, old buffer
// will not be released until new buffer comes. In order to limit the memory
// usage, here set max number of pending queued buffers to 2. If it causes some
// frame drop, just increase it to 3 or 4...

// extern bool g_bProfiling;

#define g_bProfiling true

VideoEncoder::VideoEncoder(const char *name, 
        int width, int height, uint32_t pixfmt) :
    m_name(name),
    m_width(width),
    m_height(height),
    m_pixfmt(pixfmt) //,
{
    m_VideoEncoder = NULL;
    nbs = NvBufferSessionCreate();
}

VideoEncoder::~VideoEncoder()
{ 
    if (m_VideoEncoder)
        delete m_VideoEncoder;

    if (rtp_output)
        delete gst_rtp;
}

bool VideoEncoder::initialize() {
    // Create encoder
    if (!createVideoEncoder())
        printf("Could not create encoder.");

    // Stream on
    if (m_VideoEncoder->output_plane.setStreamStatus(true) < 0)
        printf("Failed to stream on output plane");
    if (m_VideoEncoder->capture_plane.setStreamStatus(true) < 0)
        printf("Failed to stream on capture plane");

    // Set DQ callback
    m_VideoEncoder->capture_plane.setDQThreadCallback( encoderCapturePlaneDqCallback );

    // startDQThread starts a thread internally which calls the dqThreadCallback
    // whenever a buffer is dequeued on the plane
    m_VideoEncoder->capture_plane.startDQThread(this);

    // Enqueue all the empty capture plane buffers
    for (uint32_t i = 0; i < m_VideoEncoder->capture_plane.getNumBuffers(); i++) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        m_VideoEncoder->capture_plane.qBuffer(v4l2_buf, NULL);
    }

    /* Init the NvBufferTransformParams */
    memset(&m_transParams, 0, sizeof(m_transParams));
    m_transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    m_transParams.transform_filter = NvBufferTransform_Filter_Bilinear;
    m_transParams.session = nbs;
    NvBufferCreateParams cParams = {0};
    // cParams.colorFormat = NvBufferColorFormat_NV12;
    cParams.colorFormat = NvBufferColorFormat_YUV420;
    cParams.width = m_width;
    cParams.height = m_height;
    cParams.layout = NvBufferLayout_Pitch;
    cParams.payloadType = NvBufferPayload_SurfArray;
    cParams.nvbuf_tag = NvBufferTag_VIDEO_ENC;

    for(int i{0}; i < MAX_QUEUED_BUFFERS;i++) {
        if (-1 == NvBufferCreateEx(&m_que_dmabuf[i], &cParams)){
            printf("Failed to create buffers\n");    
        }
    }

    if (rtp_output){
        gst_rtp = new GstRtp();
        gst_rtp->run();
    }
    return true;
}

bool VideoEncoder::encodeFromFd(int dmabuf_fd_)
{
    int dmabuf_fd = dmabuf_fd_;
    
    if (dmabuf_fd_ > 0) {
        if (-1 == NvBufferTransform(dmabuf_fd_,  m_que_dmabuf[dmabuf_index] , &m_transParams)) 
            printf("Failed to convert the buffer to out_plane\n"); 
        dmabuf_fd = m_que_dmabuf[dmabuf_index];
    }
    else 
        dmabuf_fd = -1;

    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
    v4l2_buf.m.planes = planes;

    if (m_VideoEncoder->output_plane.getNumQueuedBuffers() < MAX_QUEUED_BUFFERS) {
        v4l2_buf.index = m_VideoEncoder->output_plane.getNumQueuedBuffers();
        v4l2_buf.m.planes[0].m.fd = dmabuf_fd;
        v4l2_buf.m.planes[0].bytesused = 1; // byteused must be non-zero
        CHECK_ERROR(m_VideoEncoder->output_plane.qBuffer(v4l2_buf, NULL));
        m_dmabufFdSet.insert(dmabuf_fd);
    }
    else {
        CHECK_ERROR(m_VideoEncoder->output_plane.dqBuffer(v4l2_buf, NULL, NULL, 10));
        // Buffer done, execute callback
        m_callback(v4l2_buf.m.planes[0].m.fd, m_callbackArg);
        m_dmabufFdSet.erase(v4l2_buf.m.planes[0].m.fd);

        if (dmabuf_fd < 0) {
            // Send EOS
            v4l2_buf.m.planes[0].bytesused = 0;
        }
        else {
            v4l2_buf.m.planes[0].m.fd = dmabuf_fd;
            v4l2_buf.m.planes[0].bytesused = 1; // byteused must be non-zero
            m_dmabufFdSet.insert(dmabuf_fd);
        }
        CHECK_ERROR(m_VideoEncoder->output_plane.qBuffer(v4l2_buf, NULL));
    }

    dmabuf_index++;
    if(dmabuf_index == MAX_QUEUED_BUFFERS) 
        dmabuf_index = 0;
    
    return true;
}

bool VideoEncoder::shutdown()
{
    // Wait till capture plane DQ Thread finishes
    // i.e. all the capture plane buffers are dequeued
    m_VideoEncoder->capture_plane.waitForDQThread(2000);

    // Return all queued buffers in output plane
    assert(m_dmabufFdSet.size() == MAX_QUEUED_BUFFERS - 1); // EOS buffer
                                                            // is not in the set
    for (std::set<int>::iterator it = m_dmabufFdSet.begin();
            it != m_dmabufFdSet.end(); it++)
    {
        m_callback(*it, m_callbackArg);
    }
    m_dmabufFdSet.clear();

    // Print profiling result
    if (g_bProfiling)
        m_VideoEncoder->printProfilingStats(std::cout);

    if (m_VideoEncoder)
    {
        delete m_VideoEncoder;
        m_VideoEncoder = NULL;
    }

    return false;
}

bool VideoEncoder::createVideoEncoder()
{
    int ret = 0;

    m_VideoEncoder = NvVideoEncoder::createVideoEncoder(m_name.c_str(), O_RDWR  /* | O_NONBLOCK */);
    if (!m_VideoEncoder)
        printf("Could not create m_VideoEncoderoder");

    // Enable profiing
    if (g_bProfiling)
        m_VideoEncoder->enableProfiling();

    ret = m_VideoEncoder->setCapturePlaneFormat(m_pixfmt, m_width,m_height,
                        8 * 1024 * 1024);
                        // 2 * 1024 * 1024);

    
    if (ret < 0)
        printf("Could not set capture plane format");

    ret = m_VideoEncoder->setOutputPlaneFormat(V4L2_PIX_FMT_NV12M, m_width, //  V4L2_PIX_FMT_YUV420M
                                    m_height);
    if (ret < 0)
        printf("Could not set output plane format");


    m_VideoEncoder->setInsertSpsPpsAtIdrEnabled(true);

    // ret = m_VideoEncoder->setBitrate(4 * 1024 * 1024);
    ret = m_VideoEncoder->setBitrate(4 * 1024 * 1024);
    if (ret < 0)
        printf("Could not set bitrate");

    if (m_pixfmt == V4L2_PIX_FMT_H264)
    {
        ret = m_VideoEncoder->setProfile(V4L2_MPEG_VIDEO_H264_PROFILE_HIGH);
        // ret = m_VideoEncoder->setProfile(V4L2_MPEG_VIDEO_H264_PROFILE_MAIN); 
    }
    else
    {
        ret = m_VideoEncoder->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN);
    }
    if (ret < 0)
        printf("Could not set m_VideoEncoderoder profile");

    if (m_pixfmt == V4L2_PIX_FMT_H264)
    {
        ret = m_VideoEncoder->setLevel(V4L2_MPEG_VIDEO_H264_LEVEL_5_0);
        
        if (ret < 0)
            printf("Could not set m_VideoEncoderoder level");
    }

    ret = m_VideoEncoder->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_CBR);
    if (ret < 0)
        printf("Could not set rate control mode");

    ret = m_VideoEncoder->setIFrameInterval(40);
    if (ret < 0)
        printf("Could not set I-frame interval");

    ret = m_VideoEncoder->setFrameRate(30, 1); // 30 0 1
    if (ret < 0)
        printf("Could not set m_VideoEncoderoder framerate");

    // Query, Export and Map the output plane buffers so that we can read
    // raw data into the buffers
    ret = m_VideoEncoder->output_plane.setupPlane(V4L2_MEMORY_DMABUF, 10, true, false);
    if (ret < 0)
        printf("Could not setup output plane");

    // Query, Export and Map the capture plane buffers so that we can write
    // encoded data from the buffers
    ret = m_VideoEncoder->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 6, true, false);
    if (ret < 0)
        printf("Could not setup capture plane");

    
    
    return true;
}

bool
VideoEncoder::encoderCapturePlaneDqCallback(struct v4l2_buffer *v4l2_buf, NvBuffer *buffer, NvBuffer *shared_buffer)
{
    if (!v4l2_buf) {
        m_VideoEncoder->abort();
        printf("Failed to dequeue buffer from capture plane");
    }

    if (rtp_output) 
        gst_rtp->setData((char *) buffer->planes[0].data, buffer->planes[0].bytesused);
    
    m_VideoEncoder->capture_plane.qBuffer(*v4l2_buf, NULL);

    // GOT EOS from encoder. Stop dqthread.
    if (buffer->planes[0].bytesused == 0) {
        return false;
    }

    return true;
}

