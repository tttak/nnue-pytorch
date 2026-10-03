"""Experimental Complex grouping only; reduction kernels remain in feature_transformer.

Fixed-size GPU metadata avoids nonzero/unique length synchronization. A workspace
is leased per CUDA device/stream/shape; same-stream queued consumers finish before
the next clear. No shared cross-stream scratch and no CPU tensor scalar reads.
"""
import torch
import cupy as cp
import threading
from contextlib import contextmanager

TIMING={scope:{stage:[] for stage in ('prepare','clear','count','scan','cursor_setup','scatter','grouped_kernel')} for scope in ('Main','FM')}
_WORKSPACES={}
_KERNELS={}
_LOCKS={}
_REGISTRY_LOCK=threading.Lock()
_ACTIVE_WORKSPACES={}
_COMPACT_KERNELS={}

_COMPACT_CODE=r'''
extern "C" __global__ void compact_active(const int* counts,int* ids,int* total,int F) {
    int f=blockIdx.x*blockDim.x+threadIdx.x;
    bool active=f<F && counts[f]>0;
    unsigned mask=__ballot_sync(0xffffffff,active);
    int base=0;
    int lane=threadIdx.x & 31;
    if(lane==0 && mask)base=atomicAdd(total,__popc(mask));
    base=__shfl_sync(0xffffffff,base,0);
    if(active)ids[base+__popc(mask & ((1u<<lane)-1u))]=f;
}
'''

def make_scheduled_reducer(M,width,fm,mode):
    from ft_count_prefix_scheduled import make_reducer
    return make_reducer(M,width,fm,mode)

def compact_active(counts,scope,timing=False):
    device=counts.device;F=counts.numel()
    key=(device.index,torch.cuda.current_stream(device).cuda_stream,F)
    if key not in _ACTIVE_WORKSPACES:
        _ACTIVE_WORKSPACES[key]=(torch.empty(F,device=device,dtype=torch.int32),torch.empty(1,device=device,dtype=torch.int32))
    ids,total=_ACTIVE_WORKSPACES[key]
    _stage(scope,'active_clear',timing,lambda:total.zero_())
    with _REGISTRY_LOCK:
        if device.index not in _COMPACT_KERNELS:
            with cp.cuda.Device(device.index):
                kernel=cp.RawKernel(_COMPACT_CODE,'compact_active');kernel.compile()
                _COMPACT_KERNELS[device.index]=kernel
        kernel=_COMPACT_KERNELS[device.index]
    with stream_context(device):
        _stage(scope,'active_compact',timing,lambda:kernel(((F+255)//256,),(256,),(counts.data_ptr(),ids.data_ptr(),total.data_ptr(),F)))
    return ids,total

@contextmanager
def workspace_scope(device):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('experimental count_prefix workspace does not support CUDA graph capture')
    key=(device.index,torch.cuda.current_stream(device).cuda_stream)
    with _REGISTRY_LOCK:
        lock=_LOCKS.setdefault(key,threading.RLock())
    # Protect enqueue order across CPU threads; never synchronize the GPU.
    with lock:yield

_CODE=r'''
extern "C" __global__ void count_features(const int* a,const int* b,int* counts,
    int n,int F,int clamp_high) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=2*n)return;
    int f=i<n?a[i]:b[i-n];
    if(f<0)return;
    if(clamp_high && f>=F)f=F-1;
    if(f<F)atomicAdd(counts+f,1);
}
extern "C" __global__ void scatter_features(const int* a,const int* b,
    const int* starts,int* cursor,int* occurrences,int n,int F,int clamp_high) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=2*n)return;
    int f=i<n?a[i]:b[i-n];
    if(f<0)return;
    if(clamp_high && f>=F)f=F-1;
    if(f<F) {
        int rank=atomicAdd(cursor+f,1);
        occurrences[starts[f]+rank]=i;
    }
}
'''

def reset_timing():
    for stages in TIMING.values():
        stages.setdefault('active_clear',[]);stages.setdefault('active_compact',[])
    for stages in TIMING.values():
        for v in stages.values():v.clear()

def stream_context(device):
    return cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream)

def _stage(scope,name,timing,operation):
    if not timing:return operation()
    with torch.profiler.record_function('NNUE/count_prefix_'+scope+'_'+name):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();result=operation();b.record();TIMING[scope].setdefault(name,[]).append((a,b))
        return result

def group(a,b,F,scope,timing=False):
    n=a.numel()
    if a.dtype!=torch.int32 or b.dtype!=torch.int32 or a.shape!=b.shape:
        raise ValueError('count_prefix needs equal int32 occurrence arrays')
    if not a.is_cuda or a.device!=b.device or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError('count_prefix needs contiguous arrays on the same CUDA device')
    if F<=0 or F>2147483647 or 2*n>2147483647:raise ValueError('int32 count/offset capacity exceeded')
    stream=torch.cuda.current_stream(a.device).cuda_stream
    key=(a.device.index,stream,F,n)
    def prepare():
        if key not in _WORKSPACES:
            _WORKSPACES[key]=dict(counts=torch.empty(F,device=a.device,dtype=torch.int32),
                ends=torch.empty(F,device=a.device,dtype=torch.int32),
                starts=torch.empty(F,device=a.device,dtype=torch.int32),
                cursor=torch.empty(F,device=a.device,dtype=torch.int32),
                features=torch.arange(F,device=a.device,dtype=torch.int32),
                occurrences=torch.empty(2*n,device=a.device,dtype=torch.int32))
        return _WORKSPACES[key]
    w=_stage(scope,'prepare',timing,prepare)
    _stage(scope,'clear',timing,lambda:w['counts'].zero_())
    with _REGISTRY_LOCK:
        if a.device.index not in _KERNELS:
            with cp.cuda.Device(a.device.index):
                kernels=(cp.RawKernel(_CODE,'count_features'),cp.RawKernel(_CODE,'scatter_features'))
                for kernel in kernels:kernel.compile()
                _KERNELS[a.device.index]=kernels
        count,scatter=_KERNELS[a.device.index]
    mode=int(scope=='FM');grid=((2*n+255)//256,);block=(256,)
    with stream_context(a.device):
        if n:
            _stage(scope,'count',timing,lambda:count(grid,block,(a.data_ptr(),b.data_ptr(),w['counts'].data_ptr(),n,F,mode)))
        _stage(scope,'scan',timing,lambda:torch.cumsum(w['counts'],dim=0,dtype=torch.int32,out=w['ends']))
        def cursor_setup():
            torch.sub(w['ends'],w['counts'],out=w['starts']);w['cursor'].zero_()
        _stage(scope,'cursor_setup',timing,cursor_setup)
        if n:
            _stage(scope,'scatter',timing,lambda:scatter(grid,block,(a.data_ptr(),b.data_ptr(),w['starts'].data_ptr(),w['cursor'].data_ptr(),w['occurrences'].data_ptr(),n,F,mode)))
    return w['occurrences'],w['features'],w['starts'],w['counts']

def workspace_bytes():
    return sum(sum(t.numel()*t.element_size() for t in w.values()) for w in _WORKSPACES.values()) + sum(sum(t.numel()*t.element_size() for t in w) for w in _ACTIVE_WORKSPACES.values())

def clear_workspace():
    # Diagnostic/lifecycle API only; caller must finish all pending GPU consumers.
    _WORKSPACES.clear()
    _ACTIVE_WORKSPACES.clear()
