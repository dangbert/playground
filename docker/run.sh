#!/bin/bash
set -e

sudo docker build . --tag gui_demo

echo -e "\nadjusting xhost permissions"
xhost si:localuser:root # allows root to user xhost
echo "starting docker..."
sudo docker container run -it --env="DISPLAY" --net=host gui_demo:latest
echo -e "\nreverting permissions..."
#xhost -si:localuser:root # revert permissions
echo "done!"