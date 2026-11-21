"""HF tokenizer.json 的轻量封装（只用 tokenizers 库，不依赖 transformers/torch）。"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tokenizers import Tokenizer

log = logging.getLogger("npu-embedding.tokenizer")


class TokenizerWrapper:
    def __init__(self, path: str, max_length: int = 512, pooling: str = "cls"):
        tok_file = path if path.endswith(".json") else os.path.join(path, "tokenizer.json")
        if not os.path.isfile(tok_file):
            raise FileNotFoundError(f"tokenizer.json not found: {tok_file}")
        self.tok = Tokenizer.from_file(tok_file)
        self.max_length = max_length
        self.pooling = pooling
        self.tok.enable_truncation(max_length=max_length)
        self.pad_id = self._resolve_pad_id(path)
        self.cls_id = self.tok.token_to_id("[CLS]")
        self.sep_id = self.tok.token_to_id("[SEP]")
        log.info("tokenizer loaded: vocab=%d pad=%d cls=%s sep=%s",
                 self.tok.get_vocab_size(), self.pad_id, self.cls_id, self.sep_id)

    def _resolve_pad_id(self, path: str) -> int:
        """优先用 tokenizer.json 的 padding 配置，其次 special_tokens_map/tokenizer_config。"""
        for name in ("tokenizer_config.json", "special_tokens_map.json"):
            cand = os.path.join(os.path.dirname(path) if path.endswith(".json") else path, name)
            try:
                with open(cand, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
            except Exception:
                continue
            pad = cfg.get("pad_token")
            if isinstance(pad, dict):
                pad = pad.get("content")
            if isinstance(pad, str):
                tid = self.tok.token_to_id(pad)
                if tid is not None:
                    return int(tid)
        tid = self.tok.token_to_id("[PAD]")
        return int(tid) if tid is not None else 0

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=True).ids

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [e.ids for e in self.tok.encode_batch(list(texts), add_special_tokens=True)]

    def build_single(self, ids: Sequence[int], seq_len: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """把单条已编码 id 序列 padding 到 seq_len，返回 OM 的三个输入张量（均 1xseq_len int64）。"""
        ids = list(ids)[:seq_len]
        n = len(ids)
        input_ids = np.full((1, seq_len), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((1, seq_len), dtype=np.int64)
        input_ids[0, :n] = ids
        attention_mask[0, :n] = 1
        token_type_ids = np.zeros((1, seq_len), dtype=np.int64)
        return input_ids, attention_mask, token_type_ids

    def build_inputs(self, texts: Sequence[str], seq_len: int
                     ) -> Tuple[Dict[str, np.ndarray], int]:
        """把一批文本编码成 OM 需要的 int64 张量，padding 到 seq_len。"""
        ids_list = self.encode_batch(texts)
        total_tokens = sum(len(x) for x in ids_list)
        n = len(ids_list)
        input_ids = np.full((n, seq_len), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((n, seq_len), dtype=np.int64)
        for i, ids in enumerate(ids_list):
            ids = ids[:seq_len]
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1
        return (
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": np.zeros((n, seq_len), dtype=np.int64),
            },
            total_tokens,
        )
