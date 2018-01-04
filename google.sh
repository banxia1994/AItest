#!/usr/bin/env sh
set -e
TOOLS=../caffe-face-caffe-face/build/tools
$TOOLS/caffe train \
    --solver=google_solver.prototxt  -gpu 0,1