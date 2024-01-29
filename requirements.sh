#!/bin/bash
# TensorFlow가 필요로 하는 버전에 맞춰 NumPy를 설치
pip install numpy==1.22

# OpenCV를 설치
pip install opencv-python==4.5.3.56

# TensorFlow를 설치
pip install tensorflow==2.12.0

# Matplotlib를 설치
pip install matplotlib

# libGL 라이브러리를 설치 (Linux)
apt-get update
apt-get install -y libgl1-mesa-glx

# 또는 headless 버전으로 설치할 수도 있음 (Linux)
apt-get install -y libgl1-mesa-glx-headless

apt-get install -y vim
