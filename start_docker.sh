#!/bin/bash

DATA_FOLDER=$1
if [ -z ${DATA_FOLDER} ]; then
    DATA_FOLDER=/home/$USER/data
fi

sudo docker build -t dreem-learning-open .
sudo docker run -it --cpuset-cpus="1-3" --gpus '"device=0"' -v "$DATA_FOLDER:/data" -v "${PWD}:/app" dreem-learning bash


