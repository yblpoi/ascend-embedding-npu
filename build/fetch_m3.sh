#!/usr/bin/env bash
set -u
cd /opt/ascend-embedding/models/hf_onnx/bge-m3
mkdir -p onnx
curl -sL --retry 3 --max-time 3600 -o onnx/model.onnx_data "https://hf-mirror.com/Xenova/bge-m3/resolve/main/onnx/model.onnx_data" 2>>/opt/ascend-embedding/build/m3dl.log
echo "m3 data rc=$? size=$(stat -c %s onnx/model.onnx_data 2>/dev/null || echo 0)" >> /opt/ascend-embedding/build/m3dl.log
curl -sL --retry 3 --max-time 600 -o onnx/model.onnx "https://hf-mirror.com/Xenova/bge-m3/resolve/main/onnx/model.onnx" 2>>/opt/ascend-embedding/build/m3dl.log
echo "m3 onnx size=$(stat -c %s onnx/model.onnx)" >> /opt/ascend-embedding/build/m3dl.log
curl -sL --max-time 300 -o sentencepiece.bpe.model "https://hf-mirror.com/Xenova/bge-m3/resolve/main/sentencepiece.bpe.model"
echo M3_DONE >> /opt/ascend-embedding/build/m3dl.log
