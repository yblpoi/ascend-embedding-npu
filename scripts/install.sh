#!/usr/bin/env bash
# 一键部署：依赖 → 模型 → ATC 转 OM → systemd 服务
# 用法: sudo bash scripts/install.sh [模型名]
# 幂等：已存在的步骤自动跳过。
set -euo pipefail

ROOT="${EMB_ROOT:-/opt/ascend-embedding}"
MODEL="${1:-bge-small-zh-v1.5}"
# BAAI 官方仓库只有 safetensors/bin，ONNX 在 Xenova 镜像仓库的 onnx/ 子目录下
HF_REPO="${HF_REPO:-Xenova/${MODEL}}"
SEQ_BUCKETS="128 256 512"
CANN_DIR="${CANN_DIR:-/usr/local/Ascend/cann-9.2.1}"

say() { printf "\033[1;36m[install]\033[0m %s\n" "$*"; }
die() { printf "\033[1;31m[install] %s\033[0m\n" "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "需要 root（sudo bash $0）"
[ -d "$CANN_DIR" ] || die "找不到 CANN: $CANN_DIR"

# NPU 运行库路径。必须显式给全：缺 plugin/opskernel 或 plugin/nnengine，
# import acl 会报 libplatform.so: undefined symbol ...protobuf7Message11GetTypeNameEv
NPC_LDP="$CANN_DIR/lib64:$CANN_DIR/lib64/plugin/opskernel:$CANN_DIR/lib64/plugin/nnengine"
NPC_LDP="$NPC_LDP:$CANN_DIR/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"
NPC_LDP="$NPC_LDP:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
NPC_LDP="$NPC_LDP:/usr/local/Ascend/nnal/asdsip/latest/lib"
# 注意：只在 import acl 检查时内联使用，不要 export 到全局。
# 全局污染会把 asdsip/driver 的库顶到最前，导致 ATC 的 TBE 初始化失败
# （Failed to initialize TeConfigInfo / OpsManager initialize failed）。

TARGET_USER="${SUDO_USER:-HwHiAiUser}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SRC_DIR/convert_om.sh" ] || die "缺少 $SRC_DIR/convert_om.sh（需与 install.sh 同目录）"
[ -f "$SRC_DIR/../deploy/ascend-embedding.service" ] || die "缺少 deploy/ascend-embedding.service"
say "安装目录=$ROOT 模型=$MODEL 运行用户=$TARGET_USER"

# ---------- 1. 目录 ----------
mkdir -p "$ROOT"/{models/hf_onnx,models/om,models/tokenizer,build,deploy,logs,scripts}
chown -R "$TARGET_USER:$TARGET_USER" "$ROOT"
say "目录就绪"

# ---------- 1.5 同步源码与脚本（从仓库根跑到 $ROOT） ----------
REPO_ROOT="$(cd "$SRC_DIR/.." && pwd)"
if [ -d "$REPO_ROOT/src/npu_embedding" ] && [ "$REPO_ROOT" != "$ROOT" ]; then
  say "同步源码与脚本到 $ROOT"
  mkdir -p "$ROOT/src" "$ROOT/deploy" "$ROOT/build"
  cp -a "$REPO_ROOT/src/npu_embedding" "$ROOT/src/"
  cp -a "$REPO_ROOT"/scripts/*.sh "$ROOT/scripts/" 2>/dev/null || true
  cp -a "$REPO_ROOT"/deploy/* "$ROOT/deploy/" 2>/dev/null || true
  cp -a "$REPO_ROOT"/build/* "$ROOT/build/" 2>/dev/null || true
  cp -a "$REPO_ROOT"/requirements.txt "$ROOT/" 2>/dev/null || true
  chown -R "$TARGET_USER:$TARGET_USER" "$ROOT"
else
  [ -d "$ROOT/src/npu_embedding" ] || die "缺少 $ROOT/src/npu_embedding（请从仓库根目录运行脚本）"
  say "源码已在 $ROOT/src，跳过同步"
fi

# ---------- 2. venv + 依赖（清华源） ----------
if [ ! -x "$ROOT/venv/bin/python" ]; then
  say "创建 venv（--system-site-packages 复用系统 numpy/onnx）"
  /usr/bin/python3 -m venv --system-site-packages "$ROOT/venv"
fi
# 依赖列表内嵌，install.sh 可单独拷走使用；仓库里有 requirements.txt 时以它为准
REQ_FILE="$(cd "$(dirname "$0")/.." && pwd)/requirements.txt"
if [ -f "$REQ_FILE" ]; then
  REQ_ARGS=(-r "$REQ_FILE")
else
  REQ_ARGS=(numpy onnx onnxruntime tokenizers "huggingface-hub" fastapi uvicorn \
            decorator attrs psutil scipy)
fi
# 系统 site-packages 里 mindstudio 等包自带过时依赖声明，pip 会刷一屏 requires 噪音
"$ROOT/venv/bin/pip" install -q --no-input --disable-pip-version-check \
  -i https://pypi.tuna.tsinghua.edu.cn/simple "${REQ_ARGS[@]}"
chown -R "$TARGET_USER:$TARGET_USER" "$ROOT/venv"
say "python 依赖就绪"

# 让 venv 自己能找到 pyACL，不依赖登录 shell 的 PYTHONPATH
PTH="$ROOT/venv/lib/python3.10/site-packages/ascend_acl.pth"
printf '%s\n%s\n' "$CANN_DIR/python/site-packages" "$ROOT/src" > "$PTH"
env -u PYTHONPATH LD_LIBRARY_PATH="$NPC_LDP" "$ROOT/venv/bin/python" -c "import acl" 2>/dev/null \
  || die "import acl 失败，检查 CANN 安装与 LD_LIBRARY_PATH"
say "pyACL 可用"

# ---------- 3. 模型（hf-mirror） ----------
ONNX_DIR="$ROOT/models/hf_onnx/$MODEL"
ONNX_FILE="$ONNX_DIR/onnx/model.onnx"
if [ ! -f "$ONNX_FILE" ]; then
  say "从 hf-mirror 下载 $HF_REPO（onnx/model.onnx + tokenizer）"
  HF_ENDPOINT=https://hf-mirror.com "$ROOT/venv/bin/python" - "$HF_REPO" "$ONNX_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
# 只取 fp32 ONNX 与 tokenizer，跳过 int8/q4/fp16 变体（省 400MB+）
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["onnx/model.onnx", "*.json", "vocab.txt",
                                  "tokenizer.json", "tokenizer_config.json"])
PY
else
  say "ONNX 已存在，跳过下载"
fi
[ -f "$ONNX_FILE" ] || die "下载后仍找不到 $ONNX_FILE"

# tokenizer（服务只用 tokenizers 库，不装 transformers）
TOK_DIR="$ROOT/models/tokenizer/$MODEL"
mkdir -p "$TOK_DIR"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json config.json vocab.txt; do
  [ -f "$ONNX_DIR/$f" ] && cp -f "$ONNX_DIR/$f" "$TOK_DIR/"
done
[ -f "$TOK_DIR/tokenizer.json" ] || die "缺少 tokenizer.json"
say "tokenizer 就绪"

# ---------- 4. ATC 转 OM ----------
OM_DIR="$ROOT/models/om/$MODEL"
mkdir -p "$OM_DIR"
for SEQ in $SEQ_BUCKETS; do
  if [ -f "$OM_DIR/seq$SEQ.om" ]; then
    say "seq$SEQ.om 已存在，跳过转换"
    continue
  fi
  say "ATC 转换 seq=$SEQ（约 1-3 分钟，内存峰值 ~12GB）"
  # ATC 在 CWD 留 fusion_result.json，先切到 build/ 再多一步清理由 convert_om.sh 负责
  bash "$SRC_DIR/convert_om.sh" "$ONNX_FILE" "$OM_DIR/seq$SEQ" "$SEQ"
done
chown -R "$TARGET_USER:$TARGET_USER" "$ROOT/models/om"
say "OM 就绪: $(ls "$OM_DIR")"

# ---------- 5. systemd ----------
say "安装 systemd 单元"
sed -e "s|^User=.*|User=$TARGET_USER|" \
    -e "s|^Group=.*|Group=$TARGET_USER|" \
    -e "s|/usr/local/Ascend/cann-9.2.1|$CANN_DIR|g" \
    -e "s|EMB_MODEL_NAME=[^\"]*|EMB_MODEL_NAME=$MODEL|" \
    -e "s|EMB_MODEL_PATH=[^\"]*|EMB_MODEL_PATH=$OM_DIR|" \
    -e "s|EMB_TOKENIZER_PATH=[^\"]*|EMB_TOKENIZER_PATH=$TOK_DIR|" \
    "$ROOT/deploy/ascend-embedding.service" \
    > /etc/systemd/system/ascend-embedding.service
systemctl daemon-reload
systemctl enable ascend-embedding
systemctl restart ascend-embedding

say "等待服务就绪..."
for i in $(seq 1 30); do
  if curl -sf --max-time 2 http://127.0.0.1:8081/health >/dev/null 2>&1; then
    say "服务已就绪"
    curl -s http://127.0.0.1:8081/health; echo
    exit 0
  fi
  sleep 2
done
die "服务 60s 内未就绪，查看: journalctl -u ascend-embedding 或 $ROOT/logs/embedding.log"