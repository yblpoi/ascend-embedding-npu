#!/usr/bin/env bash
# 把 HuggingFace ONNX 转成昇腾 .om（Ascend310B1）
# 用法: convert_om.sh <输入.onnx> <输出前缀> <seq_len>
set -euo pipefail

ONNX=$1
PREFIX=$2
SEQ=${3:-512}
ROOT="${EMB_ROOT:-/opt/ascend-embedding}"

export TE_PARALLEL_COMPILER=4          # 板子 4 核，默认 8 路并行会把内存打爆导致 ATC 假死
export ASCEND_RT_VISIBLE_DEVICES=0
# ATC 内部的 TBE/AOE 会去调 PATH 上的 python3 拿 numpy。sudo 下 PATH 是 secure_path
# （/usr/bin/python3，numpy 只装在用户 site 里），会报
#   Environment_Error_Import_Python_Module_Failed(EC0010): No module named 'numpy'
# 进而 TeConfigInfo 初始化失败、ATC run failed。把 venv/bin 顶到最前解决。
export PATH="$ROOT/venv/bin:$PATH"
source "${CANN_DIR:-/usr/local/Ascend/cann-9.2.1}/set_env.sh"

# 动态探测 ONNX 的输入名，避免硬编码
INPUTS=$(EMB_SEQ=$SEQ "$ROOT/venv/bin/python" - "$ONNX" <<'PY'
import onnx, os, sys
m = onnx.load(sys.argv[1], load_external_data=False)
seq = os.environ["EMB_SEQ"]
print(";".join(i.name + ":1," + seq for i in m.graph.input))
PY
)

echo "[convert] onnx=$ONNX out=$PREFIX seq=$SEQ"
echo "[convert] input_shape=$INPUTS"

atc \
  --model="$ONNX" \
  --framework=5 \
  --output="$PREFIX" \
  --soc_version=Ascend310B1 \
  --input_shape="$INPUTS" \
  --op_select_implmode=high_performance \
  --precision_mode=allow_fp32_to_fp16 \
  --log=error

# ATC 会在 CWD 落 fusion_result.json，别让它污染目录
rm -f "$ROOT/fusion_result.json" "$PWD/fusion_result.json" 2>/dev/null || true

echo "[convert] OK: $PREFIX.om ($(stat -c %s "$PREFIX.om") bytes)"