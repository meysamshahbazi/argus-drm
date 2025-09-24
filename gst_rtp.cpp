#include "gst_rtp.h"
#include <sstream>


GstRtp::GstRtp()
{
    m_ip_addr = "224.1.1.3";



    // clear servaddr
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_addr.s_addr = inet_addr("10.42.0.1");
    servaddr.sin_port = htons(PORT);
    servaddr.sin_family = AF_INET;
    
    // create datagram socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    
    // connect to server
    if(connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        printf("\n Error : Connect Failed \n");
        // exit(0);
    }




}

GstRtp::~GstRtp() {
    sendEos();
    while (!eos_recived) {
        usleep(200000);
    }
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
}

void GstRtp::sendEos() {
    GstFlowReturn ret;
    g_signal_emit_by_name(app_source, "end-of-stream", &ret);
}

bool GstRtp::run() {
    typedef void * (*THREADFUNCPTR)(void *);
    pthread_create(&ptid, NULL, (THREADFUNCPTR)&func, (void *)this);
}

void* GstRtp::func(void* arg) {
    pthread_detach(pthread_self());
    GstRtp* thiz = (GstRtp*)arg;
    thiz->threadFunc();
    pthread_exit(NULL);
}

void GstRtp::threadFunc() {
    create_pipe();
}

void GstRtp::create_pipe()
{
    loop = g_main_loop_new (NULL, FALSE);
    app_source = gst_element_factory_make("appsrc", "app_source");
    rtph264pay = gst_element_factory_make("rtph264pay", "rtph264_pay");
    udpsink = gst_element_factory_make("udpsink", "udp_sink");

    pipeline = gst_pipeline_new("test-pipeline");

    if (!pipeline || !app_source || !rtph264pay || !udpsink) 
        std::cout << "Not all elements could be created.\n";

    caps = gst_caps_new_simple("video/x-h264",
                    "stream-format", G_TYPE_STRING, "byte-stream",
                    "alignment", G_TYPE_STRING, "au",
                    NULL);

    g_object_set(app_source, "caps", caps, "format", GST_FORMAT_TIME, NULL);

    g_object_set(app_source,"do-timestamp", TRUE, NULL);

    // g_object_set(app_source, "caps", caps, "format", GST_FORMAT_TIME, NULL);

    gst_caps_unref(caps);

    g_object_set(rtph264pay, "pt", 96, "config-interval", 1, NULL);
    g_object_set(udpsink, "host", m_ip_addr.c_str(), "port", 5000, "sync", FALSE, NULL);

    g_signal_connect(app_source, "need-data", G_CALLBACK (start_feed), this);
    g_signal_connect(app_source, "enough-data", G_CALLBACK (stop_feed), this);

    gst_bin_add_many (GST_BIN (pipeline), app_source, rtph264pay, udpsink, NULL);
    if (gst_element_link_many(app_source, rtph264pay, udpsink, NULL) != TRUE) {
        std::cout << "Elements could not be linked.\n" << std::endl;
        gst_object_unref(pipeline);
    }

    bus = gst_element_get_bus (pipeline);
    gst_bus_add_signal_watch (bus);
    g_signal_connect (G_OBJECT (bus), "message::error", (GCallback)error_cb, this);
    gst_object_unref (bus);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    pipe_created = true;
    g_main_loop_run(loop);
}


void GstRtp::setData(char* buf, uint32_t size) {
    
    if (!pipe_created)
        return;
    
    m_buf = buf;
    m_size = size;
    push_data(this);
}

/**
 * @brief  This method is called by the idle GSource in the mainloop, to feed CHUNK_SIZE bytes into appsrc.
 * The idle handler is added to the mainloop when appsrc requests us to start sending data (need-data signal)
 * and is removed when appsrc has enough data (enough-data signal).
 * 
 * @param data 
 * @return gboolean 
 */
gboolean GstRtp::push_data(GstRtp *thiz) {
    // std::cout << "push data \n";
    GstBuffer *buffer;
    GstFlowReturn ret;
    GstMapInfo map;
    gint num_samples = thiz->m_size; 

    /* Create a new empty buffer */
    buffer = gst_buffer_new_and_alloc (thiz->m_size);
    
    // /* Set its timestamp and duration */

    gint64 ts = 0;
    g_object_get(thiz->rtph264pay, "timestamp", &ts, NULL);
    // std::cout << ts << std::endl;


    std::stringstream ss;
    ss << "ts " << ts;
    auto str = ss.str();
    sendto(thiz->sockfd, str.c_str(), MAXLINE, 0, (struct sockaddr*)NULL, sizeof(servaddr));


    // gst_util_get_timestamp();
    // GST_BUFFER_TIMESTAMP (buffer) =  gst_util_uint64_scale ((thiz->tm++), GST_SECOND, 30);
    // GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale (1, GST_SECOND, 30);

    // gst_rtp_buffer_set_timestamp (GstRTPBuffer * rtp,
    //                           guint32 timestamp)

    /* Generate some psychodelic waveforms */
    gst_buffer_map (buffer, &map, GST_MAP_WRITE);
    gint8 *raw = (gint8 *)map.data;

    for (int i = 0; i < num_samples; i++) {
        raw[i] = thiz->m_buf[i];
    }
    gst_buffer_unmap (buffer, &map);

    /* Push the buffer into the appsrc */
    g_signal_emit_by_name (thiz->app_source, "push-buffer", buffer, &ret);

    /* Free the buffer now that we are done with it */
    gst_buffer_unref (buffer);

    if (ret != GST_FLOW_OK) {
        /* We got some error, stop sending data */
        std::cout << "Error in Push Data\n";
        return FALSE;
    }
    return TRUE;
}

/**
 * @brief This signal callback triggers when appsrc needs data. Here, we add an idle handler
 * to the mainloop to start pushing data into the appsrc 
 * @param source 
 * @param size 
 * @param data 
 */
void GstRtp::start_feed(GstElement *source, guint size, GstRtp *data) {
    if (data->sourceid == 0 ) {
        std::cout << "Start Feeding\n";
        data->sourceid = g_idle_add ((GSourceFunc) push_data, data);
    }
}

/**
 * @brief  This callback triggers when appsrc has enough data and we can stop sending.
 * We remove the idle handler from the mainloop 
 * @param source 
 * @param data 
 */
void GstRtp::stop_feed(GstElement *source, GstRtp *data) {
    if (data->sourceid != 0) {
        std::cout << "Stop feeding\n";
        g_source_remove(data->sourceid);
        data->sourceid = 0;
    }
}

/**
 * @brief This function is called when an error message is posted on the bus
 * 
 * @param bus 
 * @param msg 
 * @param data 
 */
void GstRtp::error_cb(GstBus *bus, GstMessage *msg, GstRtp *data) {
    GError *err;
    gchar *debug_info;

    /* Print error details on the screen */
    gst_message_parse_error (msg, &err, &debug_info);
    g_printerr ("Error received from element %s: %s\n", GST_OBJECT_NAME (msg->src), err->message);
    g_printerr ("Debugging information: %s\n", debug_info ? debug_info : "none");
    g_clear_error (&err);
    g_free (debug_info);
}