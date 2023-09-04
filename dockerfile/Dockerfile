FROM python:3.8.10

RUN apt-get update && apt-get install -y \
    graphviz \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install jupyter

