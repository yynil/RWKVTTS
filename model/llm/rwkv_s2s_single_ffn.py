from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
from torch.utils.cpp_extension import load
import deepspeed
HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE_A", 64))
CHUNK_LEN = 16
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'{current_dir}/cuda/wkv7_cuda.cu', f'{current_dir}/cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

load(name="wkv7s", sources=[f"{current_dir}/cuda/wkv7s_op.cpp", f"{current_dir}/cuda/wkv7s.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
DTYPE = torch.bfloat16
class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
            assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.wkv7s.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y
def RWKV7_OP(state, r, w, k, v, a, b):
    return WKV_7.apply(state, r, w, k, v, a, b)

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.n_embd // self.head_size
        assert args.n_embd % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            # D_MV_LORA = 32
            if layer_id != 0:
                D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = 128
            # D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            if args.need_init_tmix:
                self._init_params(args)

    def _init_params(self, args):
        C = args.n_embd
        self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
        self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
        self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
        self.output.weight.data.zero_()

    def forward(self, x,attention_mask = None, v_first=None):
        B, T, C = x.size()
        x = x.mul(attention_mask)
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        r = r * attention_mask
        w = w * attention_mask
        k = k * attention_mask
        v = v * attention_mask
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        kk = kk * attention_mask
        k = k * (1 + (a-1) * self.k_a)
        v = v * attention_mask
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        if args.need_init_cmix:
            self._init_params(args)

    def _init_params(self, args):
        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x,attention_mask):
        x = x.mul(attention_mask)
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

    def forward(self, x,attention_mask, v_first=None):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x),attention_mask, v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x),attention_mask)
        return x, v_first
    
class L2Wrap(torch.autograd.Function):
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)
    

class RWKV7S2S_SingleFFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.n_embd % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        # 只有一套blocks，用于处理文本和音频
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        # 输出层
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.text_vocab_size, bias=False)
        self.audio_head = nn.Linear(args.n_embd, args.audio_vocab_size, bias=False)
        
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)


    def forward(self, idx, attention_mask=None, is_text=True):
        args = self.args
        B, T = idx.size()

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=idx.device)
        else:
            assert attention_mask.shape == (B, T), f'attention_mask shape: {attention_mask.shape}, idx shape: {idx.shape}'
        attention_mask = attention_mask.unsqueeze(-1)
        
        x = self.emb(idx)
        if args.dropout > 0:
            x = self.drop0(x)
        
        v_first = torch.empty_like(x)
        
        # 使用同一套blocks处理
        for i in range(args.n_layer):
            block = self.blocks[i]
            if self.training and args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, attention_mask, v_first)
            else:
                x, v_first = block(x, attention_mask, v_first)
        
        # 根据任务类型选择输出头
        if is_text:
            x = self.ln_out(x)
            text_logits = self.head(x)
            audio_logits = None
        else:
            x = self.ln_out(x)
            audio_logits = self.audio_head(x)
            text_logits = None

        return text_logits, audio_logits




# if __name__ == "__main__":
#     from argparse import Namespace
#     args = {
#         "n_layer": 24,
#         "n_embd": 1024,
#         "vocab_size": 65536,
#         "head_size_a": 64,
#         "head_size_divisor": 1,
#         "dropout": 0.1,
#         "need_init_tmix": True,
#         "need_init_cmix": True,
#         "grad_cp": 1,
#         "audio_vocab_size": 8192,
#         "text_vocab_size": 65536,
#     }
#     model = RWKV7S2S_SingleFFN(Namespace(**args))
#     print(model)
    
#     # 测试模型
#     batch_size = 2
#     seq_len = 128
#     input_ids = torch.randint(0, 65536, (batch_size, seq_len))
#     attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
#     # 测试文本模式
#     text_logits, audio_logits = model(input_ids, attention_mask, is_text=True)
#     print(f"Text mode - text_logits shape: {text_logits.shape}, audio_logits: {audio_logits}")
    
#     # 测试音频模式
#     text_logits, audio_logits = model(input_ids, attention_mask, is_text=False)
#     print(f"Audio mode - text_logits: {text_logits}, audio_logits shape: {audio_logits.shape}")
    
#     # 统计参数数量
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")


######Eval code########################################################
class RWKV_x070(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME, map_location='cuda')
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape

        keys = list(z.keys())
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
        assert self.head_size == args.head_size

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    def forward(self, idx, state, full_output=False):
        if state == None:
            state = [None for _ in range(self.n_layer * 3)]
            for i in range(self.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(self.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+1] = torch.zeros((self.n_embd // self.head_size, self.head_size, self.head_size), dtype=torch.float, requires_grad=False, device="cuda")
                state[i*3+2] = torch.zeros(self.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)

    @torch.inference_mode()
    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state
        
    @torch.inference_mode()
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state

########################################################################################################

@torch.inference_mode()
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v_first

@torch.inference_mode()
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    ######## cuda-free method 
    # w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
    # for t in range(T):
    #     r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
    #     vk = v_.view(H,N,1) @ k_.view(H,1,N)
    #     ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
    #     state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
    #     xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

    w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
    xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

    xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, x[-1,:], state, v_first

########################################################################################################

@torch.inference_mode()
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x

@torch.inference_mode()
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[-1,:]

########################################################################################################
#
# The testing code
#
########################################################################################################

@torch.inference_mode()
def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
                # assert abs(torch.sum(probs).item() - top_p) < 1e-6
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()

if __name__ == "__main__":
    from argparse import Namespace
    args = {
        "n_layer": 12,
        "n_embd": 768,
        "MODEL_NAME": "/home/yueyulin/tmp/training_0.1b/checkpoint-0-18000/pytorch_model.bin",
        "head_size": 64,
    }
    model = RWKV_x070(Namespace(**args))
    print(model)
    tokenizer_file = "/home/yueyulin/models/rwkvs2s_g1a_0.1b/rwkv_vocab_enlarged.txt"
    from tokenizer.rwkv_tokenizer import RWKV_TOKENIZER
    tokenizer = RWKV_TOKENIZER(tokenizer_file)
    str_input = "在西方，转向财政积极主义反映出人们广泛认识到货币积极主义已成强弩之末，至少边际效应已经微乎其微。"
    semantic_tokens = [7070, 4143, 1647, 6209, 4741, 2668, 4872, 3344, 4720, 7737, 6076, 3430, 3478, 7723, 8114, 4242, 6735, 1003, 121, 7463, 3198, 2795, 4286, 2844, 7850, 6698, 2451, 2668, 3653, 8170, 4563, 454, 5579, 5553, 314, 483, 3377, 5450, 5226, 2102, 6132, 6882, 8066, 7019, 3459, 5347, 4780, 8126, 3701, 5858, 5703, 832, 440, 4377, 3299, 478, 7818, 251, 6191, 2986, 2330, 4927, 3488, 2074, 1296, 426, 6719, 5423, 2669, 3789, 5648, 3874, 5690, 2409, 7091, 2815, 1364, 2622, 2897, 7907, 3909, 1334, 4737, 3268, 3887, 5713, 695, 7498, 1436, 559, 243, 2322, 5965, 249, 3317, 5103, 6954, 2207, 6465, 1169, 2605, 522, 2564, 7257, 5433, 4430, 6171, 7746, 2047, 1833, 1512, 6278, 2175, 253, 7383, 2324, 1424, 610, 874, 3244, 5793, 3132, 1484, 1731, 6884, 3131, 4347, 7958, 943, 1978, 6836, 2600, 6793, 2119, 5735, 6235, 7482, 5934, 4006, 6539, 729, 4928, 8146, 6720, 3165, 6944, 1786, 3513, 2905, 4787, 7669, 7364, 7445, 3958, 1765, 5616, 7140, 264, 3977, 7017, 7230, 2497, 2990, 4939, 7076, 138, 8153, 338, 3622, 747, 5351, 807, 936, 5875, 740, 4605, 2718, 1379, 637, 1380, 7265, 7830, 1351, 6447, 7047, 1968, 5814, 5130, 4417, 4811, 3558, 4741, 6628, 6215]

    semantic_str = ''.join([f'SEMANTIC_TOKEN_ID_{token_id}' for token_id in semantic_tokens])
    print(semantic_str)
    from utils.s2s_utilities import UNIVERSAL_ASR_TEMPLATE
    input_str = UNIVERSAL_ASR_TEMPLATE.format(SEMANTICS=semantic_str)
    print(input_str)
    input_ids = tokenizer.encode(input_str)
    input_ids = [24281, 59, 40674, 51319, 22590, 59725, 37774, 4811, 32224, 47, 11, 71052, 72036, 70457, 70697, 72068, 68988, 71869, 69340, 72048, 66409, 73005, 71356, 66060, 68015, 68485, 72691, 73105, 65882, 72701, 69917, 68295, 70787, 69663, 73301, 68712, 65765, 68252, 73569, 66986, 68671, 73671, 66837, 67918, 73603, 68070, 68554, 70087, 69274, 67624, 70725, 71910, 71063, 68566, 73310, 66825, 70152, 71789, 67985, 70395, 72258, 67134, 67890, 73333, 72016, 72300, 69634, 67689, 68505, 67310, 69813, 68549, 73294, 68185, 71103, 72179, 70348, 73513, 67842, 73420, 69561, 67776, 69206, 71971, 65880, 69433, 67546, 66579, 72934, 71527, 70159, 69756, 70786, 67917, 68698, 69434, 65679, 66071, 69305, 72070, 69748, 73242, 67664, 66141, 70023, 66374, 70523, 66668, 65662, 68146, 69026, 70836, 68791, 69087, 69530, 66224, 67257, 72738, 73158, 72919, 66401, 70294, 73563, 69970, 68954, 65842, 66667, 67787, 71447, 71355, 71117, 67559, 70241, 69883, 66220, 67348, 69752, 67781, 66833, 69829, 69695, 67593, 71327, 69371, 67809, 66243, 71640, 73441, 73633, 65821, 68567, 65787, 68570, 71905, 68526, 69127, 65914, 69804, 69609, 70979, 73691, 70047, 68292, 73675, 71150, 73771, 70693, 72292, 67294, 67651, 71832, 73334, 66725, 70982, 70531, 72081, 67256, 73412, 67140, 70670, 69799, 72991, 70414, 69713, 68308, 71138, 67364, 67026, 72779, 68986, 69391, 72563, 70706, 69931, 70589, 11, 5585,41693, 59]
    print(input_ids)
    state = None
    logits,state = model(input_ids, state)
    print(logits.shape)
    predicted_ids = []
    while True:
        idx = sample_logits(logits,top_k=20,top_p=0.95)
        if idx == 0:
            break
        predicted_ids.append(idx)
        if len(predicted_ids) > 100:
            break
        logits,state = model(idx,state)
    print(predicted_ids)
    print(tokenizer.decode(predicted_ids))