"""运行时配置（全部来自环境变量 / .env，无硬编码路径依赖）。"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List

DEFAULT_ROOT = "/opt/ascend-embedding"


def _env(key: str, default: str) -> str:
    val = os.environ.get(key)
    return default if val is None or val == "" else val


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    model_name: str
    model_path: str              # 单个 OM，或 OM 目录（多 bucket）
    tokenizer_path: str
    device_id: int = 0
    max_length: int = 512
    max_batch: int = 64
    pooling: str = "cls"         # cls | mean
    normalize: bool = True
    seq_buckets: List[int] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @classmethod
    def from_env(cls) -> "Settings":
        root = _env("EMB_ROOT", DEFAULT_ROOT)
        buckets = _env("EMB_SEQ_BUCKETS", "")
        return cls(
            model_name=_env("EMB_MODEL_NAME", "bge-small-zh-v1.5"),
            model_path=_env("EMB_MODEL_PATH", f"{root}/models/om/bge-small-zh-v1.5.om"),
            tokenizer_path=_env("EMB_TOKENIZER_PATH", f"{root}/models/tokenizer/bge-small-zh-v1.5"),
            device_id=_env_int("EMB_DEVICE_ID", 0),
            max_length=_env_int("EMB_MAX_LENGTH", 512),
            max_batch=_env_int("EMB_MAX_BATCH", 64),
            pooling=_env("EMB_POOLING", "cls").lower(),
            normalize=_env("EMB_NORMALIZE", "1") not in ("0", "false", "False"),
            seq_buckets=[int(x) for x in buckets.split(",") if x.strip().isdigit()],
        )
