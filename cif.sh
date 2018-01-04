#!/usr/bin/env sh
set -e
TOOLS=../caffe-face-caffe-face/build/tools
$TOOLS/caffe train \
    --solver=./cen_solver.prototxt -gpu 1$@ 