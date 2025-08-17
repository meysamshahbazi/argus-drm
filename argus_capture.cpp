#include "argus_capture.h"
#include <iostream>

using namespace Argus;
// using namespace ArgusSamples;

namespace ArgusSamples {
ConsumerThread::ConsumerThread(OutputStream* stream) :
        m_stream(stream),
        m_gotError(false)
{
}

ConsumerThread::~ConsumerThread()
{
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

    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
    v4l2_buf.m.planes = planes;

   

    /* Keep acquire frames and queue into encoder */
    while (!m_gotError)
    {
        NvBuffer *share_buffer;

        /* Dequeue from encoder first */
        // CHECK_ERROR(m_VideoEncoder->output_plane.dqBuffer(v4l2_buf, NULL,
        //                                                     &share_buffer, 10/*retry*/));
        /* Release the frame */
        DmaBuffer *dmabuf = static_cast<DmaBuffer*>(share_buffer);
        stream->releaseBuffer(dmabuf->getArgusBuffer());

        assert(dmabuf->getFd() == v4l2_buf.m.planes[0].m.fd);


        /* Acquire a Buffer from a completed capture request */
        Argus::Status status = STATUS_OK;
        Buffer* buffer = stream->acquireBuffer(TIMEOUT_INFINITE, &status);
        if (status == STATUS_END_OF_STREAM)
        {
            /* Timeout or error happen, exit */
            break;
        }

        std::cout << "im here\n";

        /* Convert Argus::Buffer to DmaBuffer and get FD */
        dmabuf = DmaBuffer::fromArgusBuffer(buffer);
        int dmabuf_fd = dmabuf->getFd();



    }

    requestShutdown();

    return true;
}

bool ConsumerThread::threadShutdown()
{
    return true;
}
}
ArgusCapture::ArgusCapture() {

    STREAM_SIZE = (1920, 1080);

    /* Get default EGL display */
    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDisplay == EGL_NO_DISPLAY) {
        printf("Cannot get EGL display.\n");
    }


    /* Create the CameraProvider object and get the core interface */
    cameraProvider = UniqueObj<CameraProvider>(CameraProvider::create());

    ICameraProvider *iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
    if (!iCameraProvider)
        std::cout << "ERROR Failed to create CameraProvider\n";
    
    /* Get the camera devices */
    std::vector<CameraDevice*> cameraDevices;
    iCameraProvider->getCameraDevices(&cameraDevices);
    if (cameraDevices.size() == 0)
        std::cout << "ERROR No cameras available\n";


    /* Create the capture session using the first device and get the core interface */
    UniqueObj<CaptureSession> captureSession(
            iCameraProvider->createCaptureSession(cameraDevices[CAMERA_INDEX]));
    iCaptureSession = interface_cast<ICaptureSession>(captureSession);
    if (!iCaptureSession)
        std::cout << "ERROR Failed to get ICaptureSession interface\n";

    /* Create the OutputStream */
    std::cout << "Creating output stream\n" ;
    UniqueObj<OutputStreamSettings> streamSettings(
        iCaptureSession->createOutputStreamSettings(STREAM_TYPE_BUFFER));
    IBufferOutputStreamSettings *iStreamSettings =
        interface_cast<IBufferOutputStreamSettings>(streamSettings);
    if (!iStreamSettings)
        std::cout << "ERROR Failed to get IBufferOutputStreamSettings interface\n";

    /* Configure the OutputStream to use the EGLImage BufferType */
    iStreamSettings->setBufferType(BUFFER_TYPE_EGL_IMAGE);

    /* Create the OutputStream */
    outputStream = UniqueObj<OutputStream>(iCaptureSession->createOutputStream(streamSettings.get()));
    iBufferOutputStream = interface_cast<IBufferOutputStream>(outputStream);

    /* Allocate native buffers */
    // DmaBuffer* nativeBuffers[NUM_BUFFERS];

    for (uint32_t i = 0; i < NUM_BUFFERS; i++)
    {
        nativeBuffers[i] = ArgusSamples ::DmaBuffer::create(STREAM_SIZE, NvBufferColorFormat_NV12,
                    NvBufferLayout_Pitch ); //NvBufferLayout_BlockLinear
        if (!nativeBuffers[i])
            std::cout << "ERROR Failed to allocate NativeBuffer\n";
    }

    /* Create EGLImages from the native buffers */
    // EGLImageKHR eglImages[NUM_BUFFERS];
    for (uint32_t i = 0; i < NUM_BUFFERS; i++)
    {
        eglImages[i] = nativeBuffers[i]->createEGLImage(eglDisplay);
        if (eglImages[i] == EGL_NO_IMAGE_KHR)
            std::cout << "ERROR Failed to create EGLImage\n";
    }

    /* Create the BufferSettings object to configure Buffer creation */
    UniqueObj<BufferSettings> bufferSettings(iBufferOutputStream->createBufferSettings());
    IEGLImageBufferSettings *iBufferSettings =
        interface_cast<IEGLImageBufferSettings>(bufferSettings);
    if (!iBufferSettings)
        std::cout << "ERROR Failed to create BufferSettings\n";

    /* Create the Buffers for each EGLImage (and release to
       stream for initial capture use) */
    UniqueObj<Buffer> buffers[NUM_BUFFERS];
    for (uint32_t i = 0; i < NUM_BUFFERS; i++)
    {
        iBufferSettings->setEGLImage(eglImages[i]);
        iBufferSettings->setEGLDisplay(eglDisplay);
        buffers[i].reset(iBufferOutputStream->createBuffer(bufferSettings.get()));
        IBuffer *iBuffer = interface_cast<IBuffer>(buffers[i]);

        /* Reference Argus::Buffer and DmaBuffer each other */
        iBuffer->setClientData(nativeBuffers[i]);
        nativeBuffers[i]->setArgusBuffer(buffers[i].get());

        if (!interface_cast<IEGLImageBuffer>(buffers[i]))
            std::cout << "ERROR Failed to create Buffer\n";
        if (iBufferOutputStream->releaseBuffer(buffers[i].get()) != STATUS_OK)
            std::cout << "ERROR Failed to release Buffer for capture use\n";
    }


}

bool ArgusCapture::run() {
    /* Launch the FrameConsumer thread to consume frames from the OutputStream */
    std::cout << "Launching consumer thread\n";
    ArgusSamples::ConsumerThread frameConsumerThread(outputStream.get());
    // frameConsumerThread = UniqueObj<ConsumerThread>(outputStream.get());

    frameConsumerThread.initialize();
    // PROPAGATE_ERROR();

    /* Wait until the consumer is connected to the stream */
    // PROPAGATE_ERROR(frameConsumerThread.waitRunning());
    frameConsumerThread.waitRunning();

    /* Create capture request and enable output stream */
    UniqueObj<Request> request(iCaptureSession->createRequest());
    IRequest *iRequest = interface_cast<IRequest>(request);
    if (!iRequest)
        std::cout << "ERROR Failed to create Request\n";
    iRequest->enableOutputStream(outputStream.get());

    ISourceSettings *iSourceSettings = interface_cast<ISourceSettings>(iRequest->getSourceSettings());
    if (!iSourceSettings)
        std::cout << "ERROR Failed to get ISourceSettings interface\n";
    iSourceSettings->setFrameDurationRange(Range<uint64_t>(1e9/DEFAULT_FPS));

    /* Submit capture requests */
    std::cout << "Starting repeat capture requests.\n";
    if (iCaptureSession->repeat(request.get()) != STATUS_OK)
        std::cout << "ERROR Failed to start repeat capture request\n";


    /* Wait for CAPTURE_TIME seconds */
    while ( !frameConsumerThread.isInError())
        sleep(1);

    /* Stop the repeating request and wait for idle */
    iCaptureSession->stopRepeat();
    iBufferOutputStream->endOfStream();
    iCaptureSession->waitForIdle();

    /* Wait for the consumer thread to complete */
    // PROPAGATE_ERROR(frameConsumerThread.shutdown());
    frameConsumerThread.shutdown();

    return true;
}

ArgusCapture::~ArgusCapture() {

    /* Destroy the output stream to end the consumer thread */
    outputStream.reset();

    /* Destroy the EGLImages */
    for (uint32_t i = 0; i < NUM_BUFFERS; i++)
        NvDestroyEGLImage(NULL, eglImages[i]);

    /* Destroy the native buffers */
    for (uint32_t i = 0; i < NUM_BUFFERS; i++)
        delete nativeBuffers[i];

    eglTerminate(eglDisplay);
}
