#ifndef _GST_RTP_H_
#define _GST_RTP_H_

#include <gst/gst.h>
#include <iostream>
#include <unistd.h>

// udp client driver program
#include <stdio.h>
#include <strings.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdlib.h>

#define PORT 5002
#define MAXLINE 1000




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

    void create_pipe();
    void setData(char* buf, uint32_t size);
    static void start_feed(GstElement *source, guint size, GstRtp *data);
    static void stop_feed(GstElement *source, GstRtp *data);
    static gboolean push_data(GstRtp *data);
    static void error_cb(GstBus *bus, GstMessage *msg, GstRtp *data);


    char udp_buffer[100];
    char *message = "Hello Server";
    int sockfd, n;
    struct sockaddr_in servaddr;
};

#endif

