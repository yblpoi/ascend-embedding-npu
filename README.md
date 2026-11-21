# Ascend NPU Embedding Service (香橙派 AI Pro 20T / Ascend 310B1)

在 OrangePi AI Pro 20T（昇腾 310B1，CANN 9.2.1）上部署 **bge-small-zh-v1.5** 中文 Embedding 模型，
用 **ATC 离线转成 OM**、**pyACL 直接推理**（**无 torch / 无 transformers / 无 MindSpore**），
对外提供 **OpenAI 兼容** 的 `/v1/embeddings` 接口，systemd 开机自启。

## 1. 结论先行：为什么是 bge-small-zh-v1.5

在板子上实测了 4 个候选（同一套 ATC 参数、同样 seq=512、同样 pyACL 单次同步 execute）：

| 模型 | OM 大小 | 单条延迟 (seq512) | 语义区分度 (margin) | 选型 |
|---|---|---|---|---|
| **bge-small-zh-v1.5** | 70 MB | **13.9 ms** | 0.052 | ✅ **采用** |
| bge-base-zh-v1.5 | 238 MB | 70.4 ms | 0.074 | ❌ 慢 5 倍，质量增益不足以抵消 |
| multilingual-e5-small | 429 MB | 45.5 ms | — | ❌ ONNX 有冗余节点，比 small 慢 3 倍 |
| bge-m3 | 2.3 GB | 未转换 | — | ❌ 体积/内存不适合 23GB 板子 |

**取舍逻辑**：bge-base 的语义 margin 只比 small 高 0.022（+42% 相对值），
但延迟是 5 倍、OM 体积 3.4 倍、常驻内存几乎翻倍。板上还要跑别的服务（birdid 占 ~1GB、astrbot 容器），
bge-small 是「质量够用 + 延迟最低 + 内存最省」的帕累托最优点。

**再叠加序列分档**：把同一个 ONNX 转成 seq=128/256/512 三个 OM，按输入长度选最小满足的档，
短文本（最常见：标题、短句、检索 query）延迟从 13.9 ms 降到 **1.7 ms（模型本身）**、端到端 **7.6 ms**。

## 2. 实测性能（板上）

| 场景 | 延迟 |
|---|---|
| 短句（≤128 token）端到端 HTTP | **7.6 ms** |
| 短句模型前向（seq128 OM） | 1.7 ms |
| 400 字中文（seq512 OM） | 21 ms |
| 8 并发 × 3 条批量，总耗时 | 140 ms |
| 服务常驻 RSS | ~240 MB（MemoryCurrent 773 MB 含共享） |
| 冷启动（加载 3 个 OM + tokenizer） | ~1.0 s |

OM 与 ONNXRuntime CPU 的数值一致性：**tensor cosine 0.999998+，CLS 向量 cosine 0.999999**
（fp16 精度模式，实测 4 条样本 rel err ≈ 0.0015–0.0021）。

## 3. 目录结构（板上）

```
/opt/ascend-embedding/
├── models/
│   ├── hf_onnx/bge-small-zh-v1.5/     # 原始 HF ONNX（137MB，转 OM 的源）
│   ├── tokenizer/bge-small-zh-v1.5/   # tokenizer.json 等（不装 transformers）
│   └── om/
│       └── bge-small-zh-v1.5/         # 部署用：seq128 / seq256 / seq512 三档 OM
├── src/npu_embedding/                 # 服务代码（app/config/engine/tokenizer）
├── build/                             # 转换与验证脚本 + 转换日志
├── deploy/ascend-embedding.service    # systemd 单元
├── logs/embedding.log                 # 自轮转文件日志（8MB×3）
└── venv/                              # 运行时依赖
```

占用：`models/` 292MB（OM 203MB + 原始 ONNX 89MB）、`venv/` 405MB，其余各几十 KB。

**所有产物都在 `/opt/ascend-embedding` 下**，板子根目录和家目录没有新增任何文件。
ATC 默认会在 CWD 留 `fusion_result.json`，`convert_om.sh` 里已加清理。

## 4. 从零复现

### 4.1 一键部署（推荐）

前置条件：昇腾 CANN 已装好（本项目在 `/usr/local/Ascend/cann-9.2.1` 上验证），
板子能访问 `hf-mirror.com` 与 `pypi.tuna.tsinghua.edu.cn`。

```bash
sudo git clone https://github.com/yblpoi/ascend-embedding-npu.git /opt/ascend-embedding
# 国内走代理：
#   sudo git clone https://gh-proxy.com/https://github.com/yblpoi/ascend-embedding-npu.git /opt/ascend-embedding
cd /opt/ascend-embedding
sudo bash scripts/install.sh
```

脚本按顺序做：建目录 → 同步源码 → 建 venv 装依赖（清华源）→ 写 `ascend_acl.pth`
→ hf-mirror 下 ONNX + tokenizer → ATC 转 seq128/256/512 三档 → 装 systemd 单元并启动 → 轮询 `/health`。

**幂等**：每一步都有存在性判断，重复运行只补缺的部分（已实测重跑 rc=0 全跳过）。
换模型用 `sudo bash scripts/install.sh <模型名>`（`bge-base-zh-v1.5` 也验证过能转能跑）。

**卸载**：`sudo bash scripts/uninstall.sh`（只删服务）；加 `--purge` 连 `/opt/ascend-embedding` 整个删掉。

### 4.2 手动分步（想自己控制时）

**装依赖**：

```bash
sudo mkdir -p /opt/ascend-embedding && sudo chown $USER:$USER /opt/ascend-embedding
cd /opt/ascend-embedding
/usr/bin/python3 -m venv --system-site-packages venv    # 复用系统 numpy/onnx
venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 让 venv 能找到 pyACL（不依赖登录 shell 的 PYTHONPATH）
echo /usr/local/Ascend/cann-9.2.1/python/site-packages > venv/lib/python3.10/site-packages/ascend_acl.pth
echo /opt/ascend-embedding/src >> venv/lib/python3.10/site-packages/ascend_acl.pth
```

**下模型**：注意 ONNX 在 **Xenova 镜像仓库**里，BAAI 官方仓库只有 safetensors/bin。

```bash
export HF_ENDPOINT=https://hf-mirror.com
venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
# 只取 fp32 ONNX 与 tokenizer，跳过 int8/q4/fp16 变体（省 400MB+）
snapshot_download("Xenova/bge-small-zh-v1.5",
                  local_dir="/opt/ascend-embedding/models/hf_onnx/bge-small-zh-v1.5",
                  allow_patterns=["onnx/model.onnx", "*.json", "vocab.txt",
                                  "tokenizer.json", "tokenizer_config.json"])
PY
```

**转 OM**（三档序列，短文本走短档延迟低 8 倍）：

```bash
bash scripts/convert_om.sh models/hf_onnx/bge-small-zh-v1.5/onnx/model.onnx \
     models/om/bge-small-zh-v1.5/seq128 128
# 128 / 256 / 512 各转一次，或用 build/convert_buckets.sh 一次转完
```

**装服务**：

```bash
sudo cp deploy/ascend-embedding.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ascend-embedding
```

## 5. 使用

```bash
# 健康检查
curl http://192.168.0.191:8081/health

# OpenAI 兼容调用
curl -X POST http://192.168.0.191:8081/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["今天天气真不错","昇腾 NPU 加速"],"model":"bge-small-zh-v1.5"}'
```

Python（openai SDK）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.0.191:8081/v1", api_key="not-needed")
r = client.embeddings.create(model="bge-small-zh-v1.5", input=["你好", "世界"])
print(len(r.data[0].embedding))  # 512
```

支持的参数：`input`（str / List[str]）、`dimensions`（截断后重新归一化）、
`encoding_format`（`float` / `base64`）、`model`（回显）。

### 容器内访问

服务监听 `0.0.0.0:8081`，容器两种方式都能通（已实测）：

```bash
docker run --rm --network host  curlimages/curl -s http://127.0.0.1:8081/health
docker run --rm --network bridge curlimages/curl -s http://192.168.0.191:8081/health
```

容器**不需要**设备直通——推理发生在板子进程里，容器只是 HTTP 客户端。

## 6. 运维要点（踩过的坑）

1. **systemd 必须给全 `LD_LIBRARY_PATH`**。少了 `lib64/plugin/opskernel` 或 `plugin/nnengine`，
   `import acl` 直接报 `libplatform.so: undefined symbol _ZNK14ascend_private8protobuf7Message11GetTypeNameEv`。
   完整值见 `deploy/ascend-embedding.service`。
2. **ACL 的 device/context 是线程绑定的**。FastAPI 在线程池里跑同步端点，
   每个新线程都要自己 `acl.rt.set_device()` + `acl.rt.set_context()`，
   否则第一次 memcpy 就报 `107002`。代码里用 `threading.local()` 做了 `attach_thread()`。
3. **pyACL 大多数 API 返回 tuple**：`malloc` / `load_from_file` / `create_context` / `create_stream` /
   `add_dataset_buffer` 都要解包；`get_desc` 是原地填充、返回 `None`。
4. **`acl.util.numpy_to_ptr` 在 numpy 2.x 下不可用**，一律用 `arr.ctypes.data`。
5. **nnal/atb 在这块板子上不可用**，只用了 `nnal/asdsip`；服务完全不依赖 ATB。
6. **ATC 也要 python 依赖，且必须让 `venv/bin` 排在 PATH 最前**。ATC 内部的 AOE/TBE 会通过 PATH 上的
   `python3` 导入 `numpy` / `decorator` / `attrs` / `psutil` / `scipy`。`sudo` 下 PATH 是 secure_path
   指向 `/usr/bin/python3`（这几个包只装在用户 site 里），会报
   `Environment_Error_Import_Python_Module_Failed(EC0010): No module named 'numpy'`，
   接着 `Failed to initialize TeConfigInfo` → `ATC run failed`。`convert_om.sh` 里用
   `export PATH="$ROOT/venv/bin:$PATH"` 解决，`requirements.txt` 也列了这几个包。
7. **不要在 install/convert 脚本里全局 `export LD_LIBRARY_PATH`**。给全 CANN 路径只对
   `import acl` 的检查内联使用即可；全局导出会把 asdsip/driver 的库顶到前面，同样让 ATC 的
   TBE 初始化失败——报错和上一条一模一样，很容易误判。
8. **板子的 journald 不工作**（runtime journal 从 2026-11-19 起就没再记录），
   所以额外写了 `logs/embedding.log` 自轮转文件日志。
9. `npu-smi` 显示 `Health: Alarm` 是这块板子的已知硬件设计问题，**不影响推理**。
10. 单实例约束：单元里 `MemoryMax=3G` + `TasksMax=256`，
   且 NPU 内存是设备级共享的——**绝不要起第二个实例**（会抢 3.4GB 设备内存导致 507033）。

## 7. 性能调优记录

- `--op_select_implmode=high_performance` + `allow_fp32_to_fp16` 是这里的关键：
  延迟比默认模式低约 35%，精度损失可忽略（CLS cosine 0.999999）。
- 序列分档是最大的收益点：真实流量里 >90% 是短文本，走 seq128 档，延迟降 8 倍。
- 批处理目前是「batch 内逐条 execute」：NPU 上 batch=1 的算子利用率已经不错，
  而改成动态 batch 需要重转 OM 且短文本场景收益不明显。`EMB_MAX_BATCH` 只是限流用。
- 若未来要更高吞吐，方向是：转 batch=8/16 的 OM + 服务侧攒批（当前架构已按 engine pool 抽象，加 bucket 即可）。

## 8. 目录里的脚本

| 文件 | 作用 |
|---|---|
| `scripts/convert_om.sh` | ATC 转换单个 OM |
| `build/convert_buckets.sh` | 一次转 128/256/512 三档 |
| `build/verify_om.py` | OM vs ONNXRuntime 数值一致性校验 |
| `build/bench_om.py` | 纯 pyACL 延迟基准（绕开 HTTP） |
| `build/quality_check.py` | 语义区分度（pos/neg margin）对比，用于选型 |
| `build/test_acl_infer.py` | 最小可用 pyACL 前向样例 |

## 9. 从零复现验证清单

下面是一台**已执行 `uninstall.sh --purge` 清空后**、仅用仓库内容 + `install.sh` 重建出来的实测记录
（2026-11-21，板子重启过一次，验证开机自启）：

| 检查项 | 命令 | 结果 |
|---|---|---|
| 目录清空 | `test -d /opt/ascend-embedding` | 空目录、无 uvicorn 进程 |
| 一键安装 | `sudo bash scripts/install.sh` | 下载 → 三档 ATC 全部 `ATC run success` → 服务拉起 |
| 开机自启 | `systemctl reboot` 后检查 | `is-active=active`、`is-enabled=enabled`、仅 1 个进程 |
| 健康检查 | `curl :8081/health` | `{"status":"ok","model":"bge-small-zh-v1.5","max_length":512,"buckets":[128,256,512]}` |
| 单条 | `POST /v1/embeddings` | dim 512、L2 norm 1.0、usage.prompt_tokens 正常 |
| 批量 + 截断 | `input:[3 条], dimensions:128` | n=3、每条 dim 128、仍归一化 |
| base64 | `encoding_format:"base64"` | 解出 2048 bytes（=512 float32） |
| 端到端延迟 | 10 次短句 | min 8.0 ms / p50 8.2 ms / max 12.7 ms |
| 容器 host 网络 | `docker run --rm --network host ... 127.0.0.1:8081` | 200 + 正常 JSON |
| 容器 bridge 网络 | `docker run --rm ... 192.168.0.191:8081` | 200 + 模型列表 |
| 容器内推理 | 容器内 POST /v1/embeddings | 返回正常向量 |
| 内存无泄漏 | 200 次请求前后对比 | RSS 239.9 MB → 241.4 MB（+1.4 MB），MemoryCurrent 768 MB，进程数恒为 1 |
| 卫生 | `ls /tmp`、`ls ~` | /tmp 无本项目残留；家目录项数未变（37，全为用户原有） |
| 磁盘 | `df -h /` | 46G/229G，占用 21% |

> 注意：仓库是 **private**，clone 需要 PAT：
> `sudo git clone https://<user>:<token>@github.com/yblpoi/ascend-embedding-npu.git /opt/ascend-embedding`
> 或走代理 `https://gh-proxy.com/https://<user>:<token>@github.com/...`。

安装全程（含 3 档 ATC）约 **6–10 分钟**，其中 ATC 单档 1–3 分钟、内存峰值 ~12 GB；
用 SSH 直接跑会超时，建议后台执行：
`sudo setsid bash scripts/install.sh > /tmp/inst.log 2>&1 < /dev/null &`。
