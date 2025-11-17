#include "argus_capture.h"
#include <iostream>
#include "sysio.h"

using namespace Argus;
using namespace ArgusSamples;

namespace ArgusSamples{
ConsumerThread::ConsumerThread(OutputStream* stream, ArgusCapture* ac) :
        m_stream(stream),
        ac{ac},
        m_gotError(false)
{

}

ConsumerThread::~ConsumerThread() {

}

bool ConsumerThread::threadInitialize()
{
    return true;
}

bool ConsumerThread::threadExecute() 
{
    IBufferOutputStream* stream = interface_cast<IBufferOutputStream>(m_stream);
    if (!stream)
        ORIGINATE_ERROR("Failed to get IBufferOutputStream interface");

    for (int bufferIndex = 0; bufferIndex < 1; bufferIndex++) {
        // v4l2_buf.index = bufferIndex;
        Buffer* buffer = stream->acquireBuffer();
        /* Convert Argus::Buffer to DmaBuffer and queue into v4l2 encoder */
        
        DmaBuffer *dmabuf = DmaBuffer::fromArgusBuffer(buffer);
        stream->releaseBuffer(dmabuf->getArgusBuffer());
    }

    // /* Keep acquire frames and queue into encoder */
    int idx = 0;
    while (!m_gotError) {
        NvBuffer *share_buffer;
        /* Acquire a Buffer from a completed capture request */
        Argus::Status status = STATUS_OK;
        Buffer* buffer = stream->acquireBuffer(TIMEOUT_INFINITE, &status);
        if (status == STATUS_END_OF_STREAM) {
            /* Timeout or error happen, exit */
            break;
        }
        DmaBuffer *dmabuf;
        /* Convert Argus::Buffer to DmaBuffer and get FD */
        dmabuf = DmaBuffer::fromArgusBuffer(buffer);
        int dmabuf_fd = dmabuf->getFd();
        last_fd = dmabuf_fd;
        ac->setFd(last_fd);
        
        stream->releaseBuffer(dmabuf->getArgusBuffer());
    }


    requestShutdown();
    return true;
}

bool ConsumerThread::threadShutdown()
{
    return true;
}

void ConsumerThread::abort()
{
    m_gotError = true;
}

}

ArgusCapture::ArgusCapture() {

    STREAM_SIZE = (1920, 1080);
    /* Get default EGL display */
    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
}

bool ArgusCapture::run() {
    pthread_create(&ptid_run, NULL, (THREADFUNCPTR)&func_grab_run, (void *)this);
}

void* ArgusCapture::func_grab_run(void* arg) {
    pthread_detach(pthread_self());
    ArgusCapture* thiz = (ArgusCapture*) arg;
    thiz->thread_func();
    pthread_exit(NULL);
}

bool ArgusCapture::thread_func() {
    /* Create the CameraProvider object and get the core interface */
    UniqueObj<CameraProvider> cameraProvider = UniqueObj<CameraProvider>(CameraProvider::create());
    ICameraProvider *iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
    if (!iCameraProvider)
        ORIGINATE_ERROR("Failed to create CameraProvider");

    /* Get the camera devices */
    std::vector<CameraDevice*> cameraDevices;
    iCameraProvider->getCameraDevices(&cameraDevices);
    if (cameraDevices.size() == 0)
        ORIGINATE_ERROR("No cameras available");

    if (CAMERA_INDEX >= cameraDevices.size())
    {
        PRODUCER_PRINT("CAMERA_INDEX out of range. Fall back to 0\n");
        CAMERA_INDEX = 0;
    }

    /* Create the capture session using the first device and get the core interface */
    UniqueObj<CaptureSession> captureSession(
            iCameraProvider->createCaptureSession(cameraDevices[CAMERA_INDEX]));
    ICaptureSession *iCaptureSession = interface_cast<ICaptureSession>(captureSession);
    if (!iCaptureSession)
        ORIGINATE_ERROR("Failed to get ICaptureSession interface");

    /* Create the OutputStream */
    PRODUCER_PRINT("Creating output stream\n");
    UniqueObj<OutputStreamSettings> streamSettings(
        iCaptureSession->createOutputStreamSettings(STREAM_TYPE_BUFFER));
    IBufferOutputStreamSettings *iStreamSettings =
        interface_cast<IBufferOutputStreamSettings>(streamSettings);
    if (!iStreamSettings)
        ORIGINATE_ERROR("Failed to get IBufferOutputStreamSettings interface");
    
    
    /* Configure the OutputStream to use the EGLImage BufferType */
    iStreamSettings->setBufferType(BUFFER_TYPE_EGL_IMAGE);
    // iStreamSettings->setPixelForma()
    /* Create the OutputStream */
    UniqueObj<OutputStream> outputStream(iCaptureSession->createOutputStream(streamSettings.get()));
    IBufferOutputStream *iBufferOutputStream = interface_cast<IBufferOutputStream>(outputStream);

    /* Allocate native buffers */
    DmaBuffer* nativeBuffers[NUM_BUFFERS];

    for (uint32_t i = 0; i < NUM_BUFFERS; i++) {
        nativeBuffers[i] = DmaBuffer::create(STREAM_SIZE, NvBufferColorFormat_NV12, NvBufferLayout_Pitch);
        // nativeBuffers[i] = DmaBuffer::create(STREAM_SIZE, NvBufferColorFormat_ABGR32, NvBufferLayout_Pitch);

        if (!nativeBuffers[i])
            ORIGINATE_ERROR("Failed to allocate NativeBuffer");
    }

    /* Create EGLImages from the native buffers */
    EGLImageKHR eglImages[NUM_BUFFERS];
    for (uint32_t i = 0; i < NUM_BUFFERS; i++)
    {
        eglImages[i] = nativeBuffers[i]->createEGLImage(eglDisplay);
        if (eglImages[i] == EGL_NO_IMAGE_KHR)
            ORIGINATE_ERROR("Failed to create EGLImage");
    }

    /* Create the BufferSettings object to configure Buffer creation */
    UniqueObj<BufferSettings> bufferSettings(iBufferOutputStream->createBufferSettings());
    IEGLImageBufferSettings *iBufferSettings =
        interface_cast<IEGLImageBufferSettings>(bufferSettings);
    if (!iBufferSettings)
        ORIGINATE_ERROR("Failed to create BufferSettings");

    /* Create the Buffers for each EGLImage (and release to
       stream for initial capture use) */
    UniqueObj<Buffer> buffers[NUM_BUFFERS];
    for (uint32_t i = 0; i < NUM_BUFFERS; i++)
    {
        iBufferSettings->setEGLImage(eglImages[i]);
        // iBufferSettings->set
        iBufferSettings->setEGLDisplay(eglDisplay);
        buffers[i].reset(iBufferOutputStream->createBuffer(bufferSettings.get()));
        IBuffer *iBuffer = interface_cast<IBuffer>(buffers[i]);

        /* Reference Argus::Buffer and DmaBuffer each other */
        iBuffer->setClientData(nativeBuffers[i]);
        nativeBuffers[i]->setArgusBuffer(buffers[i].get());

        if (!interface_cast<IEGLImageBuffer>(buffers[i]))
            ORIGINATE_ERROR("Failed to create Buffer");
        if (iBufferOutputStream->releaseBuffer(buffers[i].get()) != STATUS_OK)
            ORIGINATE_ERROR("Failed to release Buffer for capture use");
    }

    /* Launch the FrameConsumer thread to consume frames from the OutputStream */
    PRODUCER_PRINT("Launching consumer thread\n");
    ConsumerThread frameConsumerThread(outputStream.get(), this);
    PROPAGATE_ERROR(frameConsumerThread.initialize());

    /* Wait until the consumer is connected to the stream */
    PROPAGATE_ERROR(frameConsumerThread.waitRunning());

    /* Create capture request and enable output stream */
    UniqueObj<Request> request(iCaptureSession->createRequest());
    IRequest *iRequest = interface_cast<IRequest>(request);
    if (!iRequest)
        ORIGINATE_ERROR("Failed to create Request");
    iRequest->enableOutputStream(outputStream.get());
    // iRequest->se
    ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
    if (!iSourceSettings)
        ORIGINATE_ERROR("Failed to get ISourceSettings interface");
    iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/DEFAULT_FPS));

    /* Submit capture requests */
    PRODUCER_PRINT("Starting repeat capture requests.\n");
    if (iCaptureSession->repeat(request.get()) != STATUS_OK)
        ORIGINATE_ERROR("Failed to start repeat capture request");

    /* Wait for CAPTURE_TIME seconds */
    while (!frameConsumerThread.isInError()) {
        sleep(1);
    } 
        
    /* Stop the repeating request and wait for idle */
    iCaptureSession->stopRepeat();
    iBufferOutputStream->endOfStream();
    iCaptureSession->waitForIdle();

    /* Wait for the consumer thread to complete */
    PROPAGATE_ERROR(frameConsumerThread.shutdown());

    /* Destroy the output stream to end the consumer thread */
    outputStream.reset();

    /* Destroy the EGLImages */
    for (uint32_t i = 0; i < NUM_BUFFERS; i++)
        NvDestroyEGLImage(NULL, eglImages[i]);

    /* Destroy the native buffers */
    for (uint32_t i = 0; i < NUM_BUFFERS; i++)
        delete nativeBuffers[i];

    PRODUCER_PRINT("Done -- exiting.\n");
    return true;
}

ArgusCapture::~ArgusCapture() {
    eglTerminate(eglDisplay);
}


