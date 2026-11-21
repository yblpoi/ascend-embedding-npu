"""
NPU Embedding 服务 — OpenAI 兼容 API
基于 pyACL + 昇腾 310B1 OM 模型，无 torch 依赖。
"""
from __future__ import annotations

import base64
import json
import logging.handlers
import logging
import math
import os
import struct
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Iterable, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from npu_embedding.config import Settings
from npu_embedding.engine import NpuEnginePool
from npu_embedding.tokenizer import TokenizerWrapper

def _setup_logging() -> None:
    """stdout 给 systemd/journald，同时写一份自轮转的文件日志（journald 不可用时也能排查）。"""
    level = os.environ.get("EMB_LOG_LEVEL", "INFO").upper()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    log_file = os.environ.get("EMB_LOG_FILE", "").strip()
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=8 * 1024 * 1024, backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception as exc:  # 日志落盘失败不影响服务
            sys.stderr.write(f"file logging disabled: {exc}\n")
    # uvicorn 自带 logger 也跟随 root
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


_setup_logging()
log = logging.getLogger("npu-embedding")

settings = Settings.from_env()
pool: NpuEnginePool | None = None
tokenizer: TokenizerWrapper | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, tokenizer
    log.info("loading tokenizer from %s", settings.tokenizer_path)
    tokenizer = TokenizerWrapper(settings.tokenizer_path, settings.max_length, settings.pooling)
    log.info("loading NPU engine: model=%s device=%s", settings.model_path, settings.device_id)
    t0 = time.time()
    pool = NpuEnginePool(settings)
    log.info("engine ready in %.2fs (buckets=%s)", time.time() - t0, sorted(pool.engines))
    yield
    log.info("shutting down")
    if pool is not None:
        pool.close()


app = FastAPI(title="Ascend NPU Embedding", version="1.0.0", lifespan=lifespan)


def _b64_float32(vec: Iterable[float]) -> str:
    arr = np.asarray(list(vec), dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode()


class EmbeddingRequest(BaseModel):
    input: Any
    model: Optional[str] = None
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Any


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


def _norm_texts(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, int):
                out.append(str(item))
            elif isinstance(item, list):
                # pre-tokenized ids
                out.append(item)
            else:
                raise HTTPException(status_code=400, detail="invalid input element type")
        return out
    if isinstance(raw, int):
        return [str(raw)]
    raise HTTPException(status_code=400, detail="input must be string or array")


@app.get("/health")
def health():
    if pool is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="not ready")
    return {
        "status": "ok",
        "model": settings.model_name,
        "max_length": settings.max_length,
        "buckets": sorted(pool.engines),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": int(settings.started_at),
                "owned_by": "local-ascend-npu",
            }
        ],
    }


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    texts = _norm_texts(req.input)
    if not texts:
        raise HTTPException(status_code=400, detail="input cannot be empty")
    if len(texts) > settings.max_batch:
        raise HTTPException(status_code=400, detail=f"batch too large (max {settings.max_batch})")

    t0 = time.perf_counter()
    vectors, n_tokens = pool.embed(tokenizer, texts)
    elapsed = (time.perf_counter() - t0) * 1000

    if req.dimensions is not None and req.dimensions < vectors.shape[1]:
        vectors = vectors[:, : req.dimensions]
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    if req.encoding_format == "base64":
        payload = [_b64_float32(v) for v in vectors]
    else:
        payload = [[float(x) for x in v] for v in vectors]

    log.debug("embed n=%d tokens=%d %.1fms", len(texts), n_tokens, elapsed)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(payload)
        ],
        "model": req.model or settings.model_name,
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    }


@app.get("/")
def root():
    return {
        "service": "ascend-npu-embedding",
        "model": settings.model_name,
        "openai_compatible": "/v1/embeddings",
        "docs": "/docs",
    }