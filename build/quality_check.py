"""语义质量对比：bge-small 与 bge-base 在同一批句子上的区分度。"""
import sys, numpy as np
sys.path.insert(0, "/opt/ascend-embedding/src")
from npu_embedding.config import Settings
from npu_embedding.engine import NpuEnginePool
from npu_embedding.tokenizer import TokenizerWrapper

PAIRS = [
    ("苹果公司发布了新款手机", "iPhone 新机型正式亮相", 1),
    ("今天下大雨了", "外面雨很大，出门要带伞", 1),
    ("昇腾 NPU 可以做推理加速", "华为的神经网络处理器用于加速计算", 1),
    ("今天下大雨了", "股票市场今日大涨", 0),
    ("苹果公司发布了新款手机", "我喜欢吃红烧肉", 0),
    ("向量数据库用于相似度检索", "通过嵌入向量找最相近的文本", 1),
    ("向量数据库用于相似度检索", "汽车的刹车片需要定期更换", 0),
]

def run(model_path, tok_path, name):
    st = Settings.from_env()
    st.model_path = model_path; st.tokenizer_path = tok_path
    pool = NpuEnginePool(st)
    tk = TokenizerWrapper(tok_path, 512, "cls")
    sims, lat = [], []
    import time
    for a, b, label in PAIRS:
        t0 = time.perf_counter()
        v, _ = pool.embed(tk, [a, b])
        lat.append((time.perf_counter() - t0) * 1000)
        sims.append(float(v[0] @ v[1]))
    pos = [s for s, (_, _, l) in zip(sims, PAIRS) if l == 1]
    neg = [s for s, (_, _, l) in zip(sims, PAIRS) if l == 0]
    print("%-22s pos_mean=%.4f neg_mean=%.4f margin=%.4f min_pos=%.4f max_neg=%.4f lat2txt=%.1fms" % (
        name, np.mean(pos), np.mean(neg), min(pos) - max(neg), min(pos), max(neg), np.mean(lat)))
    for (a, b, l), s in zip(PAIRS, sims):
        print("    %s %.4f  %s | %s" % ("POS" if l else "NEG", s, a, b))
    pool.close()

if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else "/opt/ascend-embedding/models/om/bge-small-zh-v1.5"
    tp = sys.argv[2] if len(sys.argv) > 2 else "/opt/ascend-embedding/models/tokenizer/bge-small-zh-v1.5"
    nm = sys.argv[3] if len(sys.argv) > 3 else "bge-small-zh-v1.5"
    run(mp, tp, nm)