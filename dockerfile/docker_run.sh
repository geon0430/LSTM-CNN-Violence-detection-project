#!bin/bash

docker run -it  \
	--gpus all \
	--restart always \
	--shm-size=2g \
	--name geon-fastAPI_study \
	--device = /dev/video0 \
	-p 40001:8888 \
	-v /home/ \
	lstm_project
