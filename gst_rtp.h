#ifndef _GST_RTP_H_
#define _GST_RTP_H_

#include <gst/gst.h>
#include <iostream>
#include <vector>
#include <unistd.h>


class GstRtp {
private:
    GstElement *pipeline,*app_source,*rtph264pay,*udpsink;
    GstCaps *caps;
    GstBus *bus;
    guint sourceid;

    GMainLoop *loop;
    
    char *m_buf;
    uint32_t m_size;

    std::string m_ip_addr;
    bool eos_recived{false};
    bool pipe_created{false};

    pthread_t ptid;
    int tm {0};
    void sendEos();
    uint32_t frame_cnt = 0;
public:
    GstRtp();
    ~GstRtp();
    bool run();
    static void* func(void* arg);
    void threadFunc();
    void setFrameCnt(uint32_t frame_cnt);
    void create_pipe();
    void setData(char* buf, uint32_t size);
    static void start_feed(GstElement *source, guint size, GstRtp *data);
    static void stop_feed(GstElement *source, GstRtp *data);
    static gboolean push_data(GstRtp *data);
    static void error_cb(GstBus *bus, GstMessage *msg, GstRtp *data);
};

#endif

