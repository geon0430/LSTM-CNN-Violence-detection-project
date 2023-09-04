#!bin/bash

xhost +
docker run -it  \
	--gpus all \
	--restart always \
	--shm-size=2g \
	--name geon-test \
	--device /dev/video0 \
	-e DISPLAY=$DISPALY \
	-v /tmp/.X11-unix/:/tmp/.X11-unix \
	--net=host \
	--runtime nvidia \
	-p 50001:8888 \
	-v /home/geon/ \
	lstm_project:latest
