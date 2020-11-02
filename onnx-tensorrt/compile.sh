#!/bin/bash
rm -r build
mkdir build
cd build
cmake -DTENSORRT_ROOT=/media/zw/DL/ly/software/TensorRT-5.1.5.0 -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/local/bin/protoc ..
make -j4

