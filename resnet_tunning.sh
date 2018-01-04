#!/usr/bin/env sh
set -e
TOOLS=../caffe-face-caffe-face/build/tools
$TOOLS/caffe train \
    --solver=resnet_solver.prototxt -weights=resnet152_places365.caffemodel -gpu 0,1$@ 