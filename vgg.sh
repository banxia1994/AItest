#!/usr/bin/env sh
set -e
TOOLS=../caffe-face-caffe-face/build/tools
$TOOLS/caffe train \
    --solver=vgg_solver.prototxt  -gpu 0,1$@ 