#!/usr/bin/env bash
# 把本机已部署的 ascend-embedding 服务注册成 AstrBot 的 OpenAI Embedding Provider。
#
# 只需要在宿主机上跑，不需要进容器：AstrBot 把数据目录挂载到宿主机，
# 本脚本直接改宿主机上的 JSON，再重启容器生效。
#
# 用法：
#   sudo bash scripts/attach-astrbot.sh            # 自动探测
#   sudo bash scripts/attach-astrbot.sh --dry-run  # 只看会改什么
#
# 可用环境变量覆盖：
#   ASTRBOT_DATA      AstrBot 数据目录（默认 /home/HwHiAiUser/AstrBot/data）
#   ASTRBOT_CONTAINER AstrBot 容器名（默认 astrbot；设为空则不改动/重启容器）
#   EMB_API_BASE      覆盖 API Base（默认自动取 docker 网关 + 8081）
#   EMB_PROVIDER_ID   provider id（默认 ascend-npu-embedding）
#   EMB_MODEL_NAME    模型名（默认 bge-small-zh-v1.5）
#   EMB_DIMS          向量维度（默认 512）
set -euo pipefail

ASTRBOT_DATA="${ASTRBOT_DATA:-/home/HwHiAiUser/AstrBot/data}"
ASTRBOT_CONTAINER="${ASTRBOT_CONTAINER-astrbot}"
EMB_PROVIDER_ID="${EMB_PROVIDER_ID:-ascend-npu-embedding}"
EMB_MODEL_NAME="${EMB_MODEL_NAME:-bge-small-zh-v1.5}"
EMB_DIMS="${EMB_DIMS:-512}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

log() { printf '\033[1;32m[astrbot]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[astrbot]\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "请用 sudo 运行（要改 root 所有的 cmd_config.json 并重启容器）"
[ -f "$ASTRBOT_DATA/cmd_config.json" ] || die "找不到 $ASTRBOT_DATA/cmd_config.json，用 ASTRBOT_DATA 指定 AstrBot 数据目录"

# ---- 1. 决定容器侧用什么地址访问 embedding 服务 ----
if [ -z "${EMB_API_BASE:-}" ]; then
  if [ -n "$ASTRBOT_CONTAINER" ] && docker inspect "$ASTRBOT_CONTAINER" >/dev/null 2>&1; then
    # AstrBot 默认 bridge 网络：容器里的 127.0.0.1 不是宿主机，必须用 docker 网关
    GW=$(docker inspect "$ASTRBOT_CONTAINER" \
      -f '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}' 2>/dev/null | head -c 64)
    [ -n "$GW" ] || GW=172.17.0.1
    EMB_API_BASE="http://$GW:8081/v1"
  else
    EMB_API_BASE="http://127.0.0.1:8081/v1"
  fi
fi

log "AstrBot 数据目录: $ASTRBOT_DATA"
log "Embedding API Base: $EMB_API_BASE"
log "Provider id: $EMB_PROVIDER_ID / model: $EMB_MODEL_NAME / dims: $EMB_DIMS / dry-run: $DRY_RUN"

# ---- 2. 先确认这个地址真的能出向量（避免写完配置才发现连不通）----
if curl -sf -m 5 "${EMB_API_BASE%/v1}/health" >/dev/null 2>&1; then
  log "健康检查通过: ${EMB_API_BASE%/v1}/health"
else
  log "警告：${EMB_API_BASE%/v1}/health 不可达；如果是从容器里访问，确认 8081 监听 0.0.0.0 且防火墙放行"
fi

# ---- 3. 打补丁（python3 标准库即可）----
python3 - "$ASTRBOT_DATA" "$EMB_PROVIDER_ID" "$EMB_API_BASE" "$EMB_MODEL_NAME" "$EMB_DIMS" "$DRY_RUN" <<'PYEOF'
import json, os, re, shutil, sys, time

data_dir, pid, api_base, model, dims, dry = sys.argv[1:7]
dry = dry == "1"
dims = int(dims)

def load(p):
    return json.load(open(p, encoding="utf-8-sig"))

def indent_of(p):
    raw = open(p, encoding="utf-8-sig").read()
    m = re.search(r'\n( +)"', raw)
    return len(m.group(1)) if m else 2

def save(p, d):
    if dry:
        print("  [dry-run] would write", p)
        return
    bak = p + ".bak." + time.strftime("%Y%m%d%H%M%S")
    shutil.copy2(p, bak)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig") as f:
        json.dump(d, f, indent=indent_of(p), ensure_ascii=False)
    os.replace(tmp, p)
    print("  备份:", bak)

# 3.1 provider 条目
cfg_path = os.path.join(data_dir, "cmd_config.json")
cfg = load(cfg_path)
providers = [p for p in cfg.get("provider", []) if p.get("id") != pid]
providers.append({
    "id": pid,
    "type": "openai_embedding",
    "provider": "openai",
    "provider_type": "embedding",
    "enable": True,
    "embedding_api_base": api_base,
    "embedding_api_key": "not-needed",
    "embedding_model": model,
    "embedding_dimensions": dims,
    # 我们的服务支持 dimensions 参数，但这里不发，避免和向量库既有维度打架
    "embedding_dimensions_mode": "never",
    "timeout": 20,
    "proxy": "",
})
cfg["provider"] = providers
save(cfg_path, cfg)
print("  provider 已注册:", pid, "->", api_base)

# 3.2 memory_companion：打开向量召回并指向该 provider
mc_path = os.path.join(data_dir, "config", "astrbot_plugin_memory_companion_config.json")
if os.path.exists(mc_path):
    mc = load(mc_path)
    mc.setdefault("retrieval", {})["embedding_enabled"] = True
    mc["retrieval"]["embedding_provider_id"] = pid
    save(mc_path, mc)
    print("  memory_companion: embedding_enabled=True, provider_id=", pid)
else:
    print("  跳过 memory_companion（插件未安装）")

# 3.3 meme_manager：只填 provider id，不动 semantic.enabled
mm_path = os.path.join(data_dir, "config", "meme_manager_config.json")
if os.path.exists(mm_path):
    mm = load(mm_path)
    mm.setdefault("semantic", {})["embedding_provider_id"] = pid
    save(mm_path, mm)
    print("  meme_manager: semantic.embedding_provider_id=", pid, "（semantic.enabled 保持不变）")
else:
    print("  跳过 meme_manager（插件未安装）")
PYEOF

# ---- 4. 重启容器让 provider 重新加载 ----
if [ "$DRY_RUN" = "1" ]; then
  log "dry-run，不重启容器"
  exit 0
fi
if [ -n "$ASTRBOT_CONTAINER" ] && docker inspect "$ASTRBOT_CONTAINER" >/dev/null 2>&1; then
  log "重启容器 $ASTRBOT_CONTAINER ..."
  docker restart "$ASTRBOT_CONTAINER" >/dev/null
  sleep 20
  if docker logs --since 2m "$ASTRBOT_CONTAINER" 2>&1 | grep -q "\[OpenAI Embedding\] $EMB_PROVIDER_ID Using API Base"; then
    log "验证通过：容器日志已出现 [OpenAI Embedding] $EMB_PROVIDER_ID Using API Base: $EMB_API_BASE"
  else
    log "警告：日志里没看到 provider 加载成功，用这个命令看：docker logs --tail 200 $ASTRBOT_CONTAINER | grep -i embedding"
  fi
else
  log "容器未运行/未指定，请手动重启 AstrBot 使其重新加载 provider"
fi
