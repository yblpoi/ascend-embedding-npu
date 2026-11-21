import acl, numpy as np, time
acl.init()
acl.rt.set_device(0)
ctx, ret = acl.rt.create_context(0)
stream, ret = acl.rt.create_stream()
OM='/opt/ascend-embedding/models/om/bge-small-zh-v1.5.om'
mid, ret = acl.mdl.load_from_file(OM)
md = acl.mdl.create_desc(); acl.mdl.get_desc(md, mid)
ni = acl.mdl.get_num_inputs(md); no = acl.mdl.get_num_outputs(md)
in_sz = [acl.mdl.get_input_size_by_index(md,i) for i in range(ni)]
out_sz = [acl.mdl.get_output_size_by_index(md,i) for i in range(no)]
print('in_sz', in_sz, 'out_sz', out_sz)
ids = np.zeros((1,512), dtype=np.int64); ids[0,0]=101; ids[0,1]=7045; ids[0,2]=872; ids[0,3]=102
am = np.zeros((1,512), dtype=np.int64); am[0,:4]=1
tt = np.zeros((1,512), dtype=np.int64)
arrs=[ids,am,tt]
d_in=[acl.rt.malloc(s,0)[0] for s in in_sz]
d_out=[acl.rt.malloc(s,0)[0] for s in out_sz]
for i,s in enumerate(in_sz):
    ret = acl.rt.memcpy(d_in[i], s, arrs[i].ctypes.data, s, 1); assert ret==0, ('memcpy',i,ret)
ids_ds = acl.mdl.create_dataset()
for i,s in enumerate(in_sz):
    b = acl.create_data_buffer(d_in[i], s)
    ids_ds, ret = acl.mdl.add_dataset_buffer(ids_ds, b)
out_ds = acl.mdl.create_dataset()
for i,s in enumerate(out_sz):
    b = acl.create_data_buffer(d_out[i], s)
    out_ds, ret = acl.mdl.add_dataset_buffer(out_ds, b)
lat=[]
for it in range(10):
    t0=time.perf_counter(); ret = acl.mdl.execute(mid, ids_ds, out_ds); t1=time.perf_counter()
    lat.append((t1-t0)*1000)
print('exec ret', ret, 'latency ms min/avg', round(min(lat),2), round(sum(lat)/len(lat),2))
buf = np.zeros(out_sz[0], dtype=np.uint8)
ret = acl.rt.memcpy(buf.ctypes.data, out_sz[0], d_out[0], out_sz[0], 2); assert ret==0
hidden = buf.view(np.float32).reshape(1,512,512)
emb = hidden[0,0]; n=float(np.linalg.norm(emb)); emb=emb/n
print('emb[:5]', np.round(emb[:5],4).tolist(), 'norm', round(n,4))
print('nonzero', int((hidden!=0).sum()), 'max', round(float(hidden.max()),3), 'min', round(float(hidden.min()),3))
acl.mdl.destroy_dataset(ids_ds); acl.mdl.destroy_dataset(out_ds)
for p in d_in+d_out: acl.rt.free(p)
acl.mdl.unload(mid); acl.rt.destroy_stream(stream); acl.rt.destroy_context(ctx); acl.rt.reset_device(0); acl.finalize()