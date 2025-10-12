
#ifndef __VIDEOENCODER_H__
#define __VIDEOENCODER_H__

#include <fstream>
#include <iostream>
#include <set>
#include <stdio.h>
#include <unistd.h>
#include "NvVideoEncoder.h"
#include "nvbuf_utils.h"
#include "gst_rtp.h"

#define MAX_QUEUED_BUFFERS (3)


class NvBuffer;

/*
 * A helper class to simplify the usage of V4l2 encoder
 * Steps to use this class
 *   (1) Create the object
 *   (2) Call setBufferDoneCallback. The callback is called to return buffer to caller
 *   (3) Call initialize
 *   (4) Feed encoder by calling encodeFromFd
 *   (5) Call shutdown
 */
class VideoEncoder
{
public:
    VideoEncoder(const char *name, int width, int height, uint32_t pixfmt = V4L2_PIX_FMT_H265);
    ~VideoEncoder();

    bool initialize();
    bool shutdown();

    // Encode API
    bool encodeFromFd(int dmabuf_fd);

    // Callbackt to return buffer
    void setBufferDoneCallback(void (*callback)(int, void*), void *arg)
    {
        m_callback = callback;
        m_callbackArg = arg;
    }

private:
    const bool rtp_output{true};
    
    GstRtp *gst_rtp;
    
    int time_out{0};
    std::string ffmpeg_cmd;
    NvVideoEncoder *m_VideoEncoder;     // The V4L2 encoder
    bool createVideoEncoder();

    static bool encoderCapturePlaneDqCallback(
            struct v4l2_buffer *v4l2_buf,
            NvBuffer *buffer,
            NvBuffer *shared_buffer,
            void *arg)
    {
        VideoEncoder *thiz = static_cast<VideoEncoder*>(arg);
        return thiz->encoderCapturePlaneDqCallback(v4l2_buf, buffer, shared_buffer);
    }

    bool encoderCapturePlaneDqCallback(
            struct v4l2_buffer *v4l2_buf,
            NvBuffer *buffer,
            NvBuffer *shared_buffer);

    std::string m_name;     // name of the encoder
    int m_width;
    int m_height;
    uint32_t m_pixfmt;
    std::set<int> m_dmabufFdSet;    // Collection to track all queued buffer
    void (*m_callback)(int, void*);        // Output plane DQ callback
    void *m_callbackArg;

    NvBufferTransformParams m_transParams;
    NvBufferSession nbs;
    int m_que_dmabuf[MAX_QUEUED_BUFFERS];
    int dmabuf_index{0};
};

#endif  // __VIDEOENCODER_H__
