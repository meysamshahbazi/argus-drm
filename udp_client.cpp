#include "udp_client.h"
#include <sstream>

UdpClient::UdpClient() {
    std::ifstream f("/home/user/rtpip.txt");

    if (!f.is_open()) {
        std::cerr << "Error opening the file!";
    }
    // std::getline(f, );
    std::getline(f, host_ip);
    f.close();
    host_ip = "10.42.0.1";
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_addr.s_addr = inet_addr(host_ip.c_str());
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

UdpClient::~UdpClient() {

}

void UdpClient::sendMetaData(PlateResult md_) {
    md = md_;
    sendResultUDP();
}


void UdpClient::sendResultUDP() {
    unsigned char msg[28];
    msg[0] = (md.frame_cnt >> 24) & 0xff;
    msg[1] = (md.frame_cnt >> 16) & 0xff;
    msg[2] = (md.frame_cnt >> 8) & 0xff;
    msg[3] = (md.frame_cnt) & 0xff;

    msg[4] = (md.x_car >> 8) & 0xff;
    msg[5] = (md.x_car) & 0xff;
    msg[6] = (md.y_car >> 8) & 0xff;
    msg[7] = (md.y_car) & 0xff;
    msg[8] = (md.w_car >> 8) & 0xff;
    msg[9] = (md.w_car) & 0xff;
    msg[10] = (md.h_car >> 8) & 0xff;
    msg[11] = (md.h_car) & 0xff;

    msg[12] = (md.x_plt >> 8) & 0xff;
    msg[13] = (md.x_plt) & 0xff;
    msg[14] = (md.y_plt >> 8) & 0xff;
    msg[15] = (md.y_plt) & 0xff;
    msg[16] = (md.w_plt >> 8) & 0xff;
    msg[17] = (md.w_plt) & 0xff;
    msg[18] = (md.h_plt >> 8) & 0xff;
    msg[19] = (md.h_plt) & 0xff;

    while (md.plate_digit.size() < 8)
        md.plate_digit.push_back(' ');

    for (int i =0; i < 8; i++)
        msg[20 + i] = md.plate_digit[i];
    
    sendto(sockfd, msg, 28, 0, (struct sockaddr*)NULL, sizeof(servaddr));
}
