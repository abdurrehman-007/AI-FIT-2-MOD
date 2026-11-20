"""Prefer native cuDNN SDPA on eligible CUDA tensors; retain the original path."""
import os
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.nn.functional import scaled_dot_product_attention


def candidate(q):
    return (os.environ.get('AIFIT_FASHN_ATTENTION','auto')!='original'
        and q.device.type=='cuda' and q.dtype in (torch.float16,torch.bfloat16))


def eligible(q,k,v):
    params=torch.backends.cuda.SDPAParams(q,k,v,None,0.,False,False)
    return torch.backends.cuda.can_use_cudnn_attention(params)


class NativeAttention:
    def __init__(self):self.disabled=set();self.cudnn_calls=0;self.original_calls=0

    def __call__(self,q,k,v):
        signature=(tuple(q.shape),tuple(k.shape),tuple(v.shape),q.dtype)
        if signature not in self.disabled and candidate(q):
            cq=ck=cv=None
            try:
                cq,ck,cv=q.contiguous(),k.contiguous(),v.contiguous()
                if eligible(cq,ck,cv):
                    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                        output=scaled_dot_product_attention(cq,ck,cv)
                    self.cudnn_calls+=1
                    return output
            except torch.cuda.OutOfMemoryError:
                pass
            except RuntimeError as error:
                if not any(s in str(error).lower() for s in ('no available kernel','not supported','no execution plans','cudnn frontend')):
                    raise
            # Release the failed kernel frame before retrying the established
            # dispatcher. No random samples or sampler steps are consumed here.
            if len(self.disabled)<64:self.disabled.add(signature)
            cq=ck=cv=None
            torch.cuda.empty_cache()
        self.original_calls+=1
        return scaled_dot_product_attention(q,k,v)


attention=NativeAttention()


def attention_stats():
    return dict(cudnn_calls=attention.cudnn_calls,original_calls=attention.original_calls,
        disabled_shapes=len(attention.disabled))
