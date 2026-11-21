"""pyACL 推理封装：进程内共享一个 ACL context，按 seq bucket 选择 OM 引擎。"""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Sequence, Tuple

import acl
import numpy as np

log = logging.getLogger("npu-embedding.engine")

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_SUCCESS = 0


class AclSession:
    """进程级 ACL 会话：init → set_device → create_context → create_stream，只做一次。

    板子上只跑一个服务进程，ACL context 复用比"每个模型一个 context"稳定得多，
    也避免多 context 抢同一块设备内存导致 507033（设备内存不足）。
    """

    _lock = threading.Lock()
    _inst = None

    def __init__(self, device_id: int = 0):
        ret = acl.init()
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.init failed: ret={ret}")
        ret = acl.rt.set_device(device_id)
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.rt.set_device({device_id}) failed: ret={ret}")
        self.device_id = device_id
        self.context, ret = acl.rt.create_context(device_id)
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.rt.create_context failed: ret={ret}")
        self.stream, ret = acl.rt.create_stream()
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.rt.create_stream failed: ret={ret}")
        # NPU 单流串行执行：所有 execute 共用一把锁，避免多线程抢同一个 stream
        self.lock = threading.Lock()
        # ACL 的 device/context 是线程绑定的：FastAPI 在线程池里跑同步端点，
        # 每个新线程必须自己 set_device + set_context，否则 memcpy 报 107002。
        self._tls = threading.local()
        log.info("ACL session ready: device=%d", device_id)

    def attach_thread(self):
        """把当前线程绑定到本会话的 device/context（线程池里的每个线程各调一次即可）。"""
        if getattr(self._tls, "ready", False):
            return
        ret = acl.rt.set_device(self.device_id)
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.rt.set_device({self.device_id}) failed in thread: ret={ret}")
        ret = acl.rt.set_context(self.context)
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.rt.set_context failed in thread: ret={ret}")
        self._tls.ready = True

    @classmethod
    def get(cls, device_id: int = 0) -> "AclSession":
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls(device_id)
            return cls._inst

    @classmethod
    def shutdown(cls):
        with cls._lock:
            inst = cls._inst
            cls._inst = None
        if inst is None:
            return
        try:
            acl.rt.destroy_stream(inst.stream)
            acl.rt.destroy_context(inst.context)
            acl.rt.reset_device(inst.device_id)
            acl.finalize()
            log.info("ACL session closed")
        except Exception as exc:  # pragma: no cover
            log.warning("ACL shutdown error: %s", exc)


class AclEngine:
    """一个 .om 模型（batch=1，固定 seq_len）。"""

    def __init__(self, om_path: str, session: AclSession):
        self.session = session
        self.om_path = om_path
        mid, ret = acl.mdl.load_from_file(om_path)
        if ret != ACL_SUCCESS or mid is None:
            raise RuntimeError(f"acl.mdl.load_from_file({om_path}) failed: ret={ret}")
        self.model_id = mid
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, mid)

        self.in_names: List[str] = []
        self.in_dims: List[List[int]] = []
        self.in_sizes: List[int] = []
        ni = acl.mdl.get_num_inputs(self.desc)
        for i in range(ni):
            self.in_names.append(acl.mdl.get_input_name_by_index(self.desc, i))
            self.in_dims.append(list(acl.mdl.get_input_dims(self.desc, i)[0]["dims"]))
            self.in_sizes.append(acl.mdl.get_input_size_by_index(self.desc, i))

        no = acl.mdl.get_num_outputs(self.desc)
        self.out_sizes: List[int] = []
        self.out_dims: List[List[int]] = []
        for i in range(no):
            self.out_sizes.append(acl.mdl.get_output_size_by_index(self.desc, i))
            self.out_dims.append(list(acl.mdl.get_output_dims(self.desc, i)[0]["dims"]))

        self.seq_len = self.in_dims[0][1]
        self.hidden = self.out_dims[0][-1]

        self.dev_in = [self._malloc(s) for s in self.in_sizes]
        self.dev_out = [self._malloc(s) for s in self.out_sizes]
        self.in_ds = acl.mdl.create_dataset()
        for i, s in enumerate(self.in_sizes):
            self.in_ds, _ = acl.mdl.add_dataset_buffer(self.in_ds, acl.create_data_buffer(self.dev_in[i], s))
        self.out_ds = acl.mdl.create_dataset()
        for i, s in enumerate(self.out_sizes):
            self.out_ds, _ = acl.mdl.add_dataset_buffer(self.out_ds, acl.create_data_buffer(self.dev_out[i], s))
        self.host_out = np.zeros(self.out_sizes[0], dtype=np.uint8)

        log.info("engine %s: seq=%d hidden=%d inputs=%s out_dims=%s",
                 os.path.basename(om_path), self.seq_len, self.hidden,
                 self.in_names, self.out_dims)

    @staticmethod
    def _malloc(size: int) -> int:
        ptr, ret = acl.rt.malloc(size, 0)
        if ret != ACL_SUCCESS:
            raise RuntimeError(f"acl.rt.malloc({size}) failed: ret={ret}")
        return ptr

    def infer(self, tensors: Sequence[np.ndarray]) -> np.ndarray:
        self.session.attach_thread()
        for i, arr in enumerate(tensors):
            if i >= len(self.in_sizes):
                break
            arr = np.ascontiguousarray(arr, dtype=np.int64)
            ret = acl.rt.memcpy(self.dev_in[i], arr.nbytes, arr.ctypes.data,
                                arr.nbytes, ACL_MEMCPY_HOST_TO_DEVICE)
            if ret != ACL_SUCCESS:
                raise RuntimeError(f"memcpy H2D failed: ret={ret}")
        with self.session.lock:
            ret = acl.mdl.execute(self.model_id, self.in_ds, self.out_ds)
            if ret != ACL_SUCCESS:
                raise RuntimeError(f"acl.mdl.execute failed: ret={ret}")
            ret = acl.rt.memcpy(self.host_out.ctypes.data, self.out_sizes[0],
                                self.dev_out[0], self.out_sizes[0], ACL_MEMCPY_DEVICE_TO_HOST)
            if ret != ACL_SUCCESS:
                raise RuntimeError(f"memcpy D2H failed: ret={ret}")
        return self.host_out.view(np.float32).reshape(self.out_dims[0]).copy()

    def close(self):
        try:
            acl.mdl.destroy_dataset(self.in_ds)
            acl.mdl.destroy_dataset(self.out_ds)
            for p in self.dev_in + self.dev_out:
                acl.rt.free(p)
            acl.mdl.unload(self.model_id)
        except Exception as exc:  # pragma: no cover
            log.warning("engine close error: %s", exc)


class NpuEnginePool:
    """按序列长度 bucket 组织引擎：短文本走短 OM，省算力也省延迟。"""

    def __init__(self, settings):
        self.settings = settings
        self.session = AclSession.get(settings.device_id)
        self.engines: Dict[int, AclEngine] = {}
        for path in self._discover(settings):
            eng = AclEngine(path, self.session)
            self.engines[eng.seq_len] = eng
        if not self.engines:
            raise RuntimeError(f"no .om found for {settings.model_path}")
        self.buckets = sorted(self.engines)
        self.max_len = self.buckets[-1]
        log.info("engine pool: buckets=%s max_len=%d", self.buckets, self.max_len)

    @staticmethod
    def _discover(settings) -> List[str]:
        p = settings.model_path
        found: List[str] = []
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".om"):
                    found.append(os.path.join(p, name))
        elif p.endswith(".om"):
            if os.path.isfile(p):
                found.append(p)
            base = p[:-3]
            for extra in settings.seq_buckets:
                cand = f"{base}-seq{extra}.om"
                if os.path.isfile(cand):
                    found.append(cand)
        return found

    def pick(self, length: int) -> AclEngine:
        for b in self.buckets:
            if length <= b:
                return self.engines[b]
        return self.engines[self.max_len]

    def embed(self, tokenizer, texts: Sequence[str]) -> Tuple[np.ndarray, int]:
        n = len(texts)
        ids_list = tokenizer.encode_batch(texts)
        token_total = sum(len(x) for x in ids_list)
        dim = next(iter(self.engines.values())).hidden
        pooled = np.zeros((n, dim), dtype=np.float32)

        for i, ids in enumerate(ids_list):
            ids = ids[: self.max_len]
            eng = self.pick(len(ids))
            inp_ids, attn, ttype = tokenizer.build_single(ids, eng.seq_len)
            hidden = eng.infer([inp_ids, attn, ttype])[0]
            pooled[i] = self._pool(hidden, attn[0])

        if self.settings.normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pooled = pooled / norms
        return pooled, token_total

    def _pool(self, hidden: np.ndarray, attn: np.ndarray) -> np.ndarray:
        if self.settings.pooling == "mean":
            mask = attn.astype(np.float32)[:, None]
            return (hidden * mask).sum(axis=0) / max(float(mask.sum()), 1e-6)
        return hidden[0]

    def close(self):
        for eng in self.engines.values():
            eng.close()
        AclSession.shutdown()