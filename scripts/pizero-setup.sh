#!/bin/bash

# Configure Swap
# sudo dphys-swapfile swapoff
# sudo vi /etc/dphys-swapfile
# CONF_SWAPSIZE=512
# sudo dphys-swapfile setup
# sudo dphys-swapfile swapon

# System dependencies
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3-pip \
python3-venv \
python3-pil

# Adafruit Dependencies
sudo apt install -y --upgrade python3-setuptools \
i2c-tools \
libgpiod-dev \
python3-libgpiod

# setup python virtual environment
python3 -m venv ~/.robotics
source ~/.robotics/bin/activate
echo 'source ~/.robotics/bin/activate' >> ~/.bashrc

# python packages
yes | pip3 install --upgrade Rpi.gpio \
adafruit-blinka \
pillow \
numpy \
traitlets \
adafruit-circuitpython-ssd1306 \
adafruit-circuitpython-pca9685 \
adafruit-circuitpython-servokit





