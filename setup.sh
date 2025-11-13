#!/bin/sh
# sudo nvpmodel -m 0
cd build
sudo make install
cd ..

sudo cp lpr.service /etc/systemd/system/
sudo cp run.sh /usr/local/bin/

sudo chmod  744 /usr/local/bin/run.sh
sudo chmod 664 /etc/systemd/system/lpr.service
sudo systemctl daemon-reload 
sudo systemctl enable lpr.service

sudo usermod -a -G dialout $USER
sudo systemctl set-default multi-user.target

cd 
sudo rm -rf argus-drm

sudo systemctl reboot


