cat /etc/nv_tegra_release
sudo dpkg -i ./arducam-nvidia-l4t-kernel-t210-4.9.253-tegra-32.7.1-20220316013810_arm64_imx477.deb  

sudo apt-get install v4l-utils
sudo apt-get install build-essential libgtk-3-dev
apt-get install libv4l-dev
sudo systemctl set-default graphical.target
sudo systemctl set-default multi-user.target 
sudo systemctl reboot
  
find ./ -type f -perm /a+x -exec ldd {} \; \
| grep so \
| sed -e '/^[^\t]/ d' \
| sed -e 's/\t//' \
| sed -e 's/.*=..//' \
| sed -e 's/ (0.*)//' \
| sort \
| uniq -c \
| sort -n


PC:
gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,payload=96,encoding-name=H264" ! queue max-size-buffers=1 ! rtph264depay ! vaapih264dec low-latency=true ! "video/x-raw(memory:VASurface), format=(string)NV12" ! vaapisink  sync=false
gst-launch-1.0 -v udpsrc uri=udp://224.1.1.3:5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! decodebin ! videoconvert! timeoverlay ! autovideosink sync=false

gst-launch-1.0 -v udpsrc uri=udp://224.1.1.3:5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! nvh264dec ! videoconvert! timeoverlay ! autovideosink sync=false

Jetson:
gst-launch-1.0 -e nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1' ! nvv4l2h264enc bitrate=8000000 insert-sps-pps=true idrinterval=30 ! h264parse ! rtph264pay config-interval=1 mtu=1400 ! udpsink host=224.1.1.3 port=5000



