#include "gst_rtp.h"
#include <sstream>
#include <bits/stdc++.h>

GstRtp::GstRtp()
{
    std::ifstream f("/home/user/rtpip.txt");

    if (!f.is_open()) {
        std::cerr << "Error opening the file!";
    }
    std::getline(f, m_ip_addr);
    f.close();
}

/**
 * gst-launch-1.0 -e nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1' ! 
 * nvv4l2h264enc bitrate=8000000 insert-sps-pps=true idrinterval=30 ! h264parse ! rtph264pay config-interval=1 mtu=1400 ! udpsink host=192.168.1.26 port=5000
*/

GstRtp::~GstRtp() {
    sendEos();
    while (!eos_recived) {
        usleep(200000);
    }
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
}

void GstRtp::setFrameCnt(uint32_t frame_cnt){
        this->frame_cnt = frame_cnt;
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
                    "alignment", G_TYPE_STRING, "nal",
                    NULL);

    g_object_set(app_source,"do-timestamp", TRUE, NULL);

    g_object_set(app_source, "caps", caps, "format", GST_FORMAT_TIME, NULL);

    gst_caps_unref(caps);

    g_object_set(rtph264pay, "pt", 96, "config-interval", 1, "mtu", 1400, NULL);
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
    buffer = gst_buffer_new_and_alloc(thiz->m_size +15);
    
    gst_buffer_map (buffer, &map, GST_MAP_WRITE);
    guint8 *raw = (guint8 *)map.data;

    // SEI NAL Unit=[Start Code]+[NAL Unit Header (1 byte)]+[NAL Unit Payload (RBSP)]
    // NAL Unit Payload (RBSP) = [SEI Payload Type]+[SEI Payload Size]+[User Data (UUID + Frame ID)]+[RBSP Trailing Bits]
    // = 4 + 1 + 2 + (4)
    raw[0] = 0x00;
    raw[1] = 0x00;
    raw[2] = 0x00;
    raw[3] = 0x01;
    // SEI NAL Unit Header Byte=0x06
    raw[4] = 0x06;
    // For User Data Unregistered, the type is 5
    raw[5] = 0x05;
    // len 
    raw[6] = 0x07;


    raw[7] = (thiz->frame_cnt >> 24) & 0xff;
    raw[8] = 0xff;
    raw[9] = (thiz->frame_cnt >> 16) & 0xff;
    raw[10] = 0xff;
    raw[11] = (thiz->frame_cnt >> 8) & 0xff;
    raw[12] = 0xff;
    raw[13] = (thiz->frame_cnt) & 0xff;
    raw[14] = 0x80; // RBSP Trailing Bits

    for (int i = 0; i < thiz->m_size; i++) {
        raw[i+15] = thiz->m_buf[i];
    }

    gst_buffer_unmap (buffer, &map);
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
