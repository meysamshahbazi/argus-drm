#pragma once

#include <stdio.h>
#include <strings.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <bits/stdc++.h>

#define PORT 5002
#define MAXLINE 1000

struct PlateResult {
    uint16_t x_car, y_car, w_car, h_car;
    uint16_t x_plt, y_plt, w_plt, h_plt;
    std::vector<char> plate_digit;
    uint32_t frame_cnt;
};

class UdpClient {
private:
    char udp_buffer[100];
    int sockfd, n;
    struct sockaddr_in servaddr;
    std::string host_ip;
    PlateResult md;
public:
    UdpClient();
    ~UdpClient();

    void sendMetaData(PlateResult md_);
    void sendResultUDP();
};


