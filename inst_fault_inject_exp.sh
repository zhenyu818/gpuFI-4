#!/bin/bash

# set cuda installation path
export CUDA_INSTALL_PATH=/usr/local/cuda

# load environment variables
source setup_environment

# compile project
make -j$(nproc)

