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

/*
   Helper class to map NvNativeBuffer to Argus::Buffer and vice versa.
   A reference to DmaBuffer will be saved as client data in each Argus::Buffer.
   Also DmaBuffer will keep a reference to corresponding Argus::Buffer.
   This class also extends NvBuffer to act as a share buffer between Argus and V4L2 encoder.
*/
using namespace Argus;

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
    explicit ConsumerThread(OutputStream* stream);
    ~ConsumerThread();

    bool isInError()
    {
        return m_gotError;
    }

private:
    /** @name Thread methods */
    /**@{*/
    virtual bool threadInitialize();
    virtual bool threadExecute();
    virtual bool threadShutdown();
    /**@}*/
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
        return -1;
    }
private:


    /* This value is tricky.
   Too small value will impact the FPS */
    static const int    NUM_BUFFERS        = 10;
    uint32_t CAMERA_INDEX = 0;
    Size2D<uint32_t> STREAM_SIZE;
    const uint32_t            DEFAULT_FPS = 30;

    UniqueObj<CameraProvider> cameraProvider;
    ICaptureSession *iCaptureSession;
    IBufferOutputStream *iBufferOutputStream;

    EGLDisplay   eglDisplay{EGL_NO_DISPLAY};
    
    UniqueObj<OutputStream> outputStream;
    EGLImageKHR eglImages[NUM_BUFFERS];
    ArgusSamples::DmaBuffer* nativeBuffers[NUM_BUFFERS];

};

