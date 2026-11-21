#!/usr/bin/env bash
# 卸载：停服务 + 删单元；加 --purge 连 /opt 下的模型和 venv 一起删
set -euo pipefail
ROOT="${EMB_ROOT:-/opt/ascend-embedding}"
[ "$(id -u)" -eq 0 ] || { echo "需要 root"; exit 1; }

systemctl stop ascend-embedding 2>/dev/null || true
systemctl disable ascend-embedding 2>/dev/null || true
rm -f /etc/systemd/system/ascend-embedding.service
systemctl daemon-reload
echo "[uninstall] 服务已移除"

if [ "${1:-}" = "--purge" ]; then
  [ -d "$ROOT" ] || { echo "[uninstall] $ROOT 不存在"; exit 0; }
  echo "[uninstall] 将删除 $ROOT（$(du -sh "$ROOT" | cut -f1)）"
  rm -rf "$ROOT"
  echo "[uninstall] 已清空"
fi
