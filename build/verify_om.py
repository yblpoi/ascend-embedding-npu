"""校验 OM 输出与 ONNXRuntime 输出是否一致（cosine + 相对误差）。"""
import sys, numpy as np, onnxruntime as ort, onnx
from tokenizers import Tokenizer
sys.path.insert(0, "/opt/ascend-embedding/src")
import acl

MODEL_DIR = sys.argv[1]
OM = sys.argv[2]
SEQ = int(sys.argv[3])

tok = Tokenizer.from_file(MODEL_DIR + "/tokenizer.json")
tok.enable_truncation(max_length=SEQ)
sess = ort.InferenceSession(MODEL_DIR + "/model.onnx", providers=["CPUExecutionProvider"])
m = onnx.load(MODEL_DIR + "/model.onnx", load_external_data=False)
names = [i.name for i in m.graph.input]
print("onnx inputs:", names)

texts = ["今天天气真不错，适合出门散步。", "昇腾 NPU 加速的向量检索服务", "artificial intelligence and neural networks", "短"]
ids_batch = [tok.encode(t, add_special_tokens=True).ids for t in texts]

# OM 单条推理
acl.init(); acl.rt.set_device(0)
ctx, _ret = acl.rt.create_context(0); stream, _ret = acl.rt.create_stream()
mid, _ret = acl.mdl.load_from_file(OM)
md = acl.mdl.create_desc(); acl.mdl.get_desc(md, mid)
ni = acl.mdl.get_num_inputs(md); no = acl.mdl.get_num_outputs(md)
in_sz = [acl.mdl.get_input_size_by_index(md,i) for i in range(ni)]
out_sz = [acl.mdl.get_output_size_by_index(md,i) for i in range(no)]
out_dims = list(acl.mdl.get_output_dims(md,0)[0]["dims"])
d_in = [acl.rt.malloc(s,0)[0] for s in in_sz]
d_out = [acl.rt.malloc(s,0)[0] for s in out_sz]
ids_ds = acl.mdl.create_dataset()
for i,s in enumerate(in_sz): ids_ds,_ = acl.mdl.add_dataset_buffer(ids_ds, acl.create_data_buffer(d_in[i], s))
out_ds = acl.mdl.create_dataset()
for i,s in enumerate(out_sz): out_ds,_ = acl.mdl.add_dataset_buffer(out_ds, acl.create_data_buffer(d_out[i], s))
host_out = np.zeros(out_sz[0], dtype=np.uint8)

cos_list = []
for t, ids in zip(texts, ids_batch):
    n = len(ids[:SEQ])
    ii = np.full((1, SEQ), 0, dtype=np.int64); ii[0,:n] = ids[:SEQ]
    am = np.zeros((1, SEQ), dtype=np.int64); am[0,:n] = 1
    tt = np.zeros((1, SEQ), dtype=np.int64)
    feeds = {"input_ids": ii, "attention_mask": am}
    if "token_type_ids" in names: feeds["token_type_ids"] = tt
    ref = sess.run(None, feeds)[0]
    arrs = [ii, am, tt][:ni]
    for i, a in enumerate(arrs):
        acl.rt.memcpy(d_in[i], a.nbytes, a.ctypes.data, a.nbytes, 1)
    acl.mdl.execute(mid, ids_ds, out_ds)
    acl.rt.memcpy(host_out.ctypes.data, out_sz[0], d_out[0], out_sz[0], 2)
    got = host_out.view(np.float32).reshape(out_dims)
    a = ref.reshape(-1).astype(np.float64); b = got.reshape(-1).astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))
    rel = float(np.linalg.norm(a-b)/np.linalg.norm(a))
    # 池化后 embedding 的 cosine（真正影响检索的指标）
    ea = ref[0,0]; eb = got[0,0]
    ecos = float(ea @ eb / (np.linalg.norm(ea)*np.linalg.norm(eb)))
    cos_list.append((t[:18], round(cos,6), round(rel,6), round(ecos,6)))
for row in cos_list: print("  text=%-20s tensor_cos=%s rel=%s cls_cos=%s" % row)
print("mean cls cos:", round(float(np.mean([r[3] for r in cos_list])),6))

for p in d_in+d_out: acl.rt.free(p)
acl.mdl.destroy_dataset(ids_ds); acl.mdl.destroy_dataset(out_ds)
acl.mdl.unload(mid); acl.rt.destroy_stream(stream); acl.rt.destroy_context(ctx)
acl.rt.reset_device(0); acl.finalize()