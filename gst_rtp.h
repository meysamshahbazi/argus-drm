#ifndef _GST_RTP_H_
#define _GST_RTP_H_

#include <gst/gst.h>
#include <iostream>
#include <vector>
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


struct PlateResult
{
    uint16_t x_car, y_car, w_car, h_car;
    uint16_t x_plt, y_plt, w_plt, h_plt;
    std::vector<char> plate_digit;
};


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
    std::string host_ip;

    bool eos_recived{false};
    bool pipe_created{false};

    pthread_t ptid;
    int tm {0};
    void sendEos();
    uint32_t frame_cnt = 0;
    PlateResult md;

    char udp_buffer[100];
    int sockfd, n;
    struct sockaddr_in servaddr;


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
    void setMetaData(PlateResult md_);

    void sendResultUDP();

   
};

#endif

