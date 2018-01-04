#!/usr/bin/env sh
set -e
TOOLS=../caffe-face-caffe-face/build/tools
$TOOLS/caffe train \
    --solver=vgg_solver.prototxt -weights=vgg16_places365.caffemodel -gpu 0,1$@ 