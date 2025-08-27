#pragma once

#include <Argus/Argus.h>
#include "Error.h"
#include "Thread.h"
#include <nvbuf_utils.h>
#include <stdio.h>
#include <stdlib.h>
#include "nvmmapi/NvNativeBuffer.h"
#include <NvApplicationProfiler.h>
#include "BufferStream.h"
#include "NvBuffer.h"
#include <NvVideoEncoder.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>


using namespace Argus;

/*
   Helper class to map NvNativeBuffer to Argus::Buffer and vice versa.
   A reference to DmaBuffer will be saved as client data in each Argus::Buffer.
   Also DmaBuffer will keep a reference to corresponding Argus::Buffer.
   This class also extends NvBuffer to act as a share buffer between Argus and V4L2 encoder.
*/


/* Debug print macros */
#define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)
#define CONSUMER_PRINT(...) printf("CONSUMER: " __VA_ARGS__)
#define CHECK_ERROR(expr) \
    do { \
        if ((expr) < 0) { \
            abort(); \
            ORIGINATE_ERROR(#expr " failed"); \
        } \
    } while (0);


/* Constant configuration */
static const int    MAX_ENCODER_FRAMES = 5;
static const int    DEFAULT_FPS        = 30;
static const int    Y_INDEX            = 0;
static const int    START_POS          = 32;
static const int    FONT_SIZE          = 64;
static const int    SHIFT_BITS         = 3;
static const int    array_n[8][8] = {
    { 1, 1, 0, 0, 0, 0, 1, 1 },
    { 1, 1, 1, 0, 0, 0, 1, 1 },
    { 1, 1, 1, 1, 0, 0, 1, 1 },
    { 1, 1, 1, 1, 1, 0, 1, 1 },
    { 1, 1, 0, 1, 1, 1, 1, 1 },
    { 1, 1, 0, 0, 1, 1, 1, 1 },
    { 1, 1, 0, 0, 0, 1, 1, 1 },
    { 1, 1, 0, 0, 0, 0, 1, 1 }
};


/* Configurations which can be overrided by cmdline */
static int          CAPTURE_TIME = 5; // In seconds.
static uint32_t     CAMERA_INDEX = 0;
static Size2D<uint32_t> STREAM_SIZE (640, 480);
static std::string  OUTPUT_FILENAME ("output.h264");
static uint32_t     ENCODER_PIXFMT = V4L2_PIX_FMT_H264;
static bool         DO_STAT = false;
static bool         VERBOSE_ENABLE = false;
static bool         DO_CPU_PROCESS = false;

static EGLDisplay   eglDisplay = EGL_NO_DISPLAY;



class ArgusCapture;

namespace ArgusSamples
{
    class DmaBuffer : public NvNativeBuffer, public NvBuffer
{
public:
    /* Always use this static method to create DmaBuffer */
    static DmaBuffer* create(const Argus::Size2D<uint32_t>& size,
                             NvBufferColorFormat colorFormat,
                             NvBufferLayout layout = NvBufferLayout_Pitch)
    {
        DmaBuffer* buffer = new DmaBuffer(size);
        if (!buffer)
            return NULL;

        if (NvBufferCreate(&buffer->m_fd, size.width(), size.height(), layout, colorFormat))
        {
            delete buffer;
            return NULL;
        }

        /* save the DMABUF fd in NvBuffer structure */
        buffer->planes[0].fd = buffer->m_fd;
        /* byteused must be non-zero for a valid buffer */
        buffer->planes[0].bytesused = 1;

        return buffer;
    }

    /* Help function to convert Argus Buffer to DmaBuffer */
    static DmaBuffer* fromArgusBuffer(Buffer *buffer)
    {
        IBuffer* iBuffer = interface_cast<IBuffer>(buffer);
        const DmaBuffer *dmabuf = static_cast<const DmaBuffer*>(iBuffer->getClientData());

        return const_cast<DmaBuffer*>(dmabuf);
    }

    /* Return DMA buffer handle */
    int getFd() const { return m_fd; }

    /* Get and set reference to Argus buffer */
    void setArgusBuffer(Buffer *buffer) { m_buffer = buffer; }
    Buffer *getArgusBuffer() const { return m_buffer; }

private:
    DmaBuffer(const Argus::Size2D<uint32_t>& size)
        : NvNativeBuffer(size),
          NvBuffer(0, 0),
          m_buffer(NULL)
    {
    }

    Buffer *m_buffer;   /* Reference to Argus::Buffer */
};


/**
 * Consumer thread:
 *   Acquire frames from BufferOutputStream and extract the DMABUF fd from it.
 *   Provide DMABUF to V4L2 for video encoding. The encoder will save the encoded
 *   stream to disk.
 */
class ConsumerThread : public Thread
{
public:
    explicit ConsumerThread(OutputStream* stream, ArgusCapture *ac);
    ~ConsumerThread();

    bool isInError()
    {
        return m_gotError;
    }
    int getFd() {
        return last_fd;
    }

private:
    int last_fd{-1};
    ArgusCapture *ac{nullptr};
    /** @name Thread methods */
    /**@{*/
    virtual bool threadInitialize();
    virtual bool threadExecute();
    virtual bool threadShutdown();
    /**@}*/
    void abort();

    OutputStream* m_stream;
    bool m_gotError;
};


}

class ArgusCapture 
{
public:
    ArgusCapture();
    ~ArgusCapture();
    bool run();
    int getFd() {
        return last_fd;
    }
    void setFd(int fd_) {
        last_fd = fd_;
    }
    bool thread_func();
    static void* func_grab_run(void* arg);
private:
    pthread_t ptid_run;
    int last_fd {-1};
    /* This value is tricky.
     Too small value will impact the FPS */
    static const int    NUM_BUFFERS        = 10;
    uint32_t CAMERA_INDEX = 0;
    Size2D<uint32_t> STREAM_SIZE;
    const uint32_t            DEFAULT_FPS = 30;

    // UniqueObj<CameraProvider> cameraProvider;
    // ICaptureSession *iCaptureSession;
    // IBufferOutputStream *iBufferOutputStream;

    // EGLDisplay   eglDisplay{EGL_NO_DISPLAY};
    
    // UniqueObj<OutputStream> outputStream;
    // EGLImageKHR eglImages[NUM_BUFFERS];
    // ArgusSamples::DmaBuffer* nativeBuffers[NUM_BUFFERS];


    // NvV4l2Element *capture_plane;

};

