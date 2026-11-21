#!/usr/bin/env bash
# 一次转出 128/256/512 三个序列档（短文本走短档，延迟低 8 倍）
# 用法: convert_buckets.sh [ONNX路径] [ONNX输出目录]
set -u
ROOT="${EMB_ROOT:-/opt/ascend-embedding}"
ONNX="${1:-$ROOT/models/hf_onnx/bge-small-zh-v1.5/model.onnx}"
OUTDIR="${2:-$ROOT/models/om/bge-small-zh-v1.5}"
LOG="$ROOT/build/atc-buckets.log"
mkdir -p "$OUTDIR"
: > "$LOG"

for SEQ in 128 256 512; do
  echo "[buckets] converting seq=$SEQ" | tee -a "$LOG"
  "$ROOT/scripts/convert_om.sh" "$ONNX" "$OUTDIR/seq${SEQ}" "$SEQ" >> "$LOG" 2>&1
  rc=$?
  echo "[buckets] seq=$SEQ rc=$rc" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[buckets] FAILED at seq=$SEQ" | tee -a "$LOG"; exit $rc; }
  # 转一个就释放一次内存，避免 4 核板子被 ATC 中间产物撑爆
  sleep 2
done
echo "[buckets] ALL DONE" | tee -a "$LOG"
ls -la "$OUTDIR"
