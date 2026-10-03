"""Exp155: same nonempty reducer arithmetic, early return / active scheduling only."""
import ast, inspect, textwrap, threading
import torch
import cupy as cp
_CACHE={}
_LOCK=threading.Lock()

def make_reducer(M,width,fm,mode):
    import feature_transformer as FT
    key=(M,width,fm,mode,torch.cuda.current_device())
    with _LOCK:
        if key in _CACHE:return _CACHE[key]
        factory=FT.make_fm_embedding_grouped_backward_kernel if fm else FT.make_double_feature_transformer_slice_backward_kernel
        tree=ast.parse(textwrap.dedent(inspect.getsource(factory)))
        literals=[n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel_code' for t in n.targets) and isinstance(n.value,ast.Constant)]
        assert len(literals)==1
        threads=32 if fm else 128
        code=literals[0].format(max_active_features=M,output_size=width,factor_dim=width,num_threads=threads)
        if mode=='count_prefix_early':
            needle='group_counts[group_id];'
            assert code.count(needle)==1
            code=code.replace(needle,needle+'\n    if (group_count == 0) return;')
        elif mode=='count_prefix_active':
            code=code.replace('const uint32_t batch_size','const int32_t* __restrict__ active_count,\n    const uint32_t batch_size')
            begin=code.index('{',code.index('extern "C"'))
            end=code.index('const int32_t feature_index',begin)
            code=code[:begin+1]+'''\n    const uint32_t tid = threadIdx.x;
    for (uint32_t active_id = blockIdx.x;
         active_id < (uint32_t)(*active_count); active_id += gridDim.x) {
    const uint32_t group_id = unique_features[active_id];
    '''+code[end:]
            code=code.replace('unique_features[group_id]','group_id')
            closing=code.rfind('}')
            code=code[:closing]+'}\n'+code[closing:]
        else:raise ValueError(mode)
        name='fm_embedding_grouped_backward' if fm else 'double_feature_transformer_slice_backward_grouped'
        kernel=cp.RawKernel(code,name);kernel.compile()
        _CACHE[key]=FT._kernel_with_threads(kernel,(threads,))
        return _CACHE[key]
