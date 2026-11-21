import acl, numpy as np, time, sys
OM = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv)>2 else 10
acl.init(); acl.rt.set_device(0)
ctx,_ = acl.rt.create_context(0); stream,_ = acl.rt.create_stream()
mid, ret = acl.mdl.load_from_file(OM)
md = acl.mdl.create_desc(); acl.mdl.get_desc(md, mid)
ni = acl.mdl.get_num_inputs(md); no = acl.mdl.get_num_outputs(md)
in_sz = [acl.mdl.get_input_size_by_index(md,i) for i in range(ni)]
out_sz = [acl.mdl.get_output_size_by_index(md,i) for i in range(no)]
dims = [acl.mdl.get_input_dims(md,i)[0]['dims'] for i in range(ni)]
print('dims', dims, 'out', acl.mdl.get_output_dims(md,0)[0]['dims'], 'out_sz', out_sz)
seq = dims[0][1]
arrs=[np.zeros((1,seq), dtype=np.int64) for _ in range(ni)]
arrs[0][0,:8]=[101,7045,872,1024,256,3399,102,0]
if ni>1: arrs[1][0,:7]=1
d_in=[acl.rt.malloc(s,0)[0] for s in in_sz]; d_out=[acl.rt.malloc(s,0)[0] for s in out_sz]
for i,s in enumerate(in_sz): acl.rt.memcpy(d_in[i], s, arrs[i].ctypes.data, s, 1)
ids_ds = acl.mdl.create_dataset()
for i,s in enumerate(in_sz): ids_ds,_ = acl.mdl.add_dataset_buffer(ids_ds, acl.create_data_buffer(d_in[i], s))
out_ds = acl.mdl.create_dataset()
for i,s in enumerate(out_sz): out_ds,_ = acl.mdl.add_dataset_buffer(out_ds, acl.create_data_buffer(d_out[i], s))
lat=[]
for it in range(N):
    t0=time.perf_counter(); acl.mdl.execute(mid, ids_ds, out_ds); acl.rt.synchronize_stream(stream); lat.append((time.perf_counter()-t0)*1000)
print('latency ms: min %.2f  p50 %.2f  mean %.2f' % (min(lat), sorted(lat)[len(lat)//2], sum(lat)/len(lat)))
acl.mdl.destroy_dataset(ids_ds); acl.mdl.destroy_dataset(out_ds)
for p in d_in+d_out: acl.rt.free(p)
acl.mdl.unload(mid); acl.rt.destroy_stream(stream); acl.rt.destroy_context(ctx); acl.rt.reset_device(0); acl.finalize()