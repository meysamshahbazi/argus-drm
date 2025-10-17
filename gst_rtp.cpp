#include "gst_rtp.h"
#include <sstream>

// #include "my_frame_id_meta.h"
#include "gst/video/gstvideometa.h"
// #include <gst/rtp/gstrtpmeta.h>

GstRtp::GstRtp()
{
    // host=192.168.1.26
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
    // g_object_set(rtph264pay, "pt", 96, /* "config-interval", 1, "mtu", 1400, */ NULL);
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

    frame_cnt++;
    if (frame_cnt == 0) // avoid 4 zeros
        frame_cnt++;

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
    
    int start_index = 6;
    int len = 0;
    
    uint8_t embed_data[len] = {31, 32, 33, 34};

    /* Create a new empty buffer */
    buffer = gst_buffer_new_and_alloc (thiz->m_size +15);
    
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
    raw[6] = 0x08;


    raw[7] = (thiz->frame_cnt >> 24) & 0xff;
    raw[8] = (thiz->frame_cnt >> 16) & 0xff;
    raw[9] = (thiz->frame_cnt >> 8) & 0xff;
    raw[10] = (thiz->frame_cnt) & 0xff;


    raw[11] = 0x00;
    raw[12] = 0x00;
    raw[13] = 0x00;
    raw[14] = 0x00;

    for (int i = 0; i < thiz->m_size; i++) {
        raw[i+15] = thiz->m_buf[i];
    }

/* 
    for (int i = 0; i < start_index; i++) {
        raw[i] = thiz->m_buf[i];
    }

    for (int j = 0; j <len; j++) {
        raw[start_index+j] = embed_data[j];
    }

    for (int i = start_index +len; i < num_samples + len; i++) {
        raw[i] = thiz->m_buf[i-len];
    }

    */



    // for (int i = 0; i < index; i++) {
    //     raw[i] = thiz->m_buf[i];
    // }

    // raw[index] = 10;
    // raw[index+1] = 11;
    // raw[index+2] = 12;
    // raw[index+3] = 13;
    // raw[index+4] = 14;

    // for (int i = index; i < num_samples; i++) {
    //     raw[i+index] = thiz->m_buf[i];
    // }
    //

    std::cout << map.size << "\t";

    // for (int i =0; i < 40; i++)
    //     std::cout << int(raw[i]) << ", ";
    
    int nb_nal = 0;
    for (int i =0; i < map.size-3; i++) {   
        if (raw[i]   == 0x00 &&
            raw[i+1] == 0x00 &&
            raw[i+2] == 0x00 &&
            raw[i+3] == 0x01 ) {
                nb_nal++; 
                std::cout << "h: " << int(raw[i+4] ) << " \t";
            }    
    }

    std::cout << nb_nal << " ";

    std::cout << std::endl;
    
    

    gst_buffer_unmap (buffer, &map);

    /* Push the buffer into the appsrc */
    // GST_BUFFER_PTS (buffer) = 0;// gst_util_get_timestamp();

    g_signal_emit_by_name (thiz->app_source, "push-buffer", buffer, &ret);

    /* Free the buffer now that we are done with it */
    gst_buffer_unref (buffer);

    // GstClockTime ts;
    // g_object_get(thiz->rtph264pay, "timestamp", &ts, NULL);
    


    // GstClockTime pts = GST_BUFFER_PTS(buffer);
    // std::cout << pts << "\n";

    if (ret != GST_FLOW_OK) {
        /* We got some error, stop sending data */
        std::cout << "Error in Push Data\n";
        return FALSE;
    }
    return TRUE;


    // guint64 current_frame_id = 1234;
    // GstVideoCropMeta *meta = (GstVideoCropMeta *)gst_buffer_add_meta(buffer, GST_VIDEO_CROP_META_INFO, NULL);
    // meta->x = current_frame_id;



    // GstMetaInfo gmi;
    // gst_buffer_add_meta(buffer, &gmi, nullptr);


    // GstBuffer *buffer;
    // GstFlowReturn ret;

    // 1. Increment the shared ID
    // global_frame_counter++;


    // 2. Prepare the video buffer
    // You should set PTS and DTS here if you are not using auto-timestamps.
    // e.g., GST_BUFFER_PTS (buffer) = gst_util_get_timestamp();

    // 3. Attach the custom metadata
    // gst_buffer_add_my_frame_id_meta (buffer, current_frame_id);
    
    // --- Synchronization Step ---
    // 4. Send the metadata (Frame ID + Results) via UDP
    
    // Get your computed results

    // Create the UDP packet: [Frame ID (8 bytes)] [x] [y] [w] [h]
    // The key is including the 'current_frame_id'
    // NOTE: Ensure your packing/endianness (e.g., struct.pack in Python, or use a custom C struct) 
    // is consistent between sender (Jetson) and receiver (PC).
    
    // Example pseudocode for sending UDP:
    // send_udp_packet_with_id(current_frame_id, x, y, w, h);
    
    // 5. Push the buffer to the GStreamer pipeline
    // ret = gst_app_src_push_buffer (appsrc, buffer);

    // if (ret != GST_FLOW_OK) {
    //     g_warning ("Push buffer failed: %s", gst_flow_return_get_name(ret));
    // }

    // /* Set its timestamp and duration */

    // gint64 ts = 0;
    


    // std::stringstream ss;
    // // ss << "ts " << ts;
    // auto str = ss.str();
    // std::cout << str << std::endl;
    // sendto(thiz->sockfd, str.c_str(), MAXLINE, 0, (struct sockaddr*)NULL, sizeof(servaddr));


    // gst_util_get_timestamp();
    // GST_BUFFER_TIMESTAMP (buffer) =  gst_util_uint64_scale ((0), GST_SECOND, 30);
    // GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale (1, GST_SECOND, 30);

    // gst_rtp_buffer_set_timestamp (GstRTPBuffer * rtp,
    //                           guint32 timestamp)

    /* Generate some psychodelic waveforms */


    // gst_buffer_add_meta(buffer, )


    // static const gchar* tags[] = { NULL };
    // auto meta_info = gst_meta_register_custom ("mymeta", tags, NULL, NULL, NULL);
    // gst_meta_api_type_register (const gchar * api,
    //                         const gchar ** tags)
    // auto meta = gst_buffer_add_custom_meta (buffer, "mymeta");
    // auto metadata = gst_custom_meta_get_structure (meta);
    // gst_structure_set (metadata, "uniq_id", G_TYPE_INT64, gint64(1234), nullptr);



    // static const gchar* tags[] = { NULL };
    // auto meta_info = gst_meta_register_custom ("mymeta", tags, NULL, NULL, NULL);
    // // gst_meta_register(tags, )

    // // gst_buffer_add_meta(buffer, const GstMetaInfo *info,
    // //                                              gpointer params);
    // auto meta = gst_buffer_add_custom_meta (buffer, "mymeta");
    // auto metadata = gst_custom_meta_get_structure (meta);
    // gst_structure_set (metadata, "property_name", G_TYPE_INT64, gint64(1), nullptr);

    
    // GstVideoCropMeta gvcm;
    
    
    

   
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