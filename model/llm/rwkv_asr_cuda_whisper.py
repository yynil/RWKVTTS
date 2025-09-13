from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import io
from typing import OrderedDict
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder
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


DTYPE = torch.bfloat16
load(name="rwkv7_state_fwd_fp16", sources=[f"{current_dir}/cuda/rwkv7_state_fwd_fp16.cpp", f"{current_dir}/cuda/rwkv7_state_fwd_fp16.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
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
            torch.ops.rwkv7_state_fwd_fp16.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y
def RWKV7_OP(state, r, w, k, v, a, b):
    return WKV_7.apply(state, r, w, k, v, a, b)

class WKV_7_batch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
            assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.rwkv7_state_fwd_fp16.forward(B, T, C, H, state, r, w, k, v, a, b, y)
            return y
def RWKV7_BATCH_OP(state, r, w, k, v, a, b):
    return WKV_7_batch.apply(state, r, w, k, v, a, b)


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

    @torch.inference_mode()
    def forward_batch(self, x,attention_mask = None, v_first=None,x_prev=None,state=None):
        B, T, C = x.size()
        x = x.mul(attention_mask)
        H = self.n_head
        xx = torch.cat((x_prev.unsqueeze(1), x[:,:-1,:]), dim=1) - x

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
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        v = v * attention_mask
        x = RWKV7_BATCH_OP(state,r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first,x[:,-1,:],state

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
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
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

    @torch.inference_mode()
    def forward_batch(self, x,attention_mask = None,x_prev=None):
        B, T, C = x.size()
        x = x.mul(attention_mask)
        xx = torch.cat((x_prev.unsqueeze(1), x[:,:-1,:]), dim=1) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k),x[:,-1,:]

    def forward(self, x,attention_mask):
        x = x.mul(attention_mask)
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    
class Block(nn.Module):
    def __init__(self, args, layer_id,attn=None):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        if attn is None:
            self.att = RWKV_Tmix_x070(args, layer_id)
        else:
            self.att = attn
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

    @torch.inference_mode()
    def forward_batch(self, x,attention_mask = None, v_first=None,tx_prev=None,state=None,cx_prev=None):
        if self.layer_id == 0:
            x = self.ln0(x)
        x_attn, v_first, tx_prev, state = self.att.forward_batch(self.ln1(x),attention_mask, v_first,tx_prev,state)
        x = x + x_attn
        x_ffn, cx_prev = self.ffn.forward_batch(self.ln2(x),attention_mask,cx_prev)
        x = x + x_ffn
        return x, v_first, tx_prev, state, cx_prev

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
    
class RWKV7ModelForLatentInputsCuda(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.n_embd % 32 == 0


        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    @torch.inference_mode()
    def forward_batch(self, latents,attention_mask = None):
        args = self.args
        B, T,C = latents.size()
        states = [None for _ in range(args.n_layer * 3)]
        for i in range(args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
            states[i*3+0] = torch.zeros((B, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            states[i*3+1] = torch.zeros((B, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
            states[i*3+2] = torch.zeros((B, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=latents.device)
        else:
            assert attention_mask.shape == (B, T), f'attention_mask shape: {attention_mask.shape}, latents shape: {latents.shape}'
        attention_mask = attention_mask.unsqueeze(-1)
        x = latents
        v_first = torch.empty_like(x)
        for i in range(args.n_layer):
            block = self.blocks[i]
            tx_prev = states[i*3+0]
            state = states[i*3+1]
            cx_prev = states[i*3+2]
            x, v_first, tx_prev, state, cx_prev = block.forward_batch(x,attention_mask, v_first,tx_prev,state,cx_prev)
            states[i*3+0] = tx_prev
            states[i*3+1] = state
            states[i*3+2] = cx_prev
        x = self.ln_out(x)
        x = x[:, :T]
        return x
    def forward(self, latents,attention_mask = None):
        args = self.args
        B, T,C = latents.size()
    
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=latents.device)
        else:
            assert attention_mask.shape == (B, T), f'attention_mask shape: {attention_mask.shape}, latents shape: {latents.shape}'
        
        if T%16 != 0:
            #right padding to 16x
            padding_length = 16 - T%16
            latents = torch.cat([latents, torch.zeros(B, padding_length, C,device=latents.device,dtype=latents.dtype)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros(B, padding_length, dtype=torch.bool, device=latents.device)], dim=1)

        attention_mask = attention_mask.unsqueeze(-1)
        x = latents
        if args.dropout > 0:
            x = self.drop0(x)
        v_first = torch.empty_like(x)
        for i in range(args.n_layer):
            block = self.blocks[i]
            if self.training and args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, attention_mask, v_first)
            else:
                x, v_first = block(x,attention_mask, v_first)
        x = self.ln_out(x)
        x = x[:, :T]
        return x


class RWKV7ModelForCausalLMCuda(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.n_embd % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    

    @torch.inference_mode()
    def forward_batch(self, embds,attention_mask = None,states = None):
        args = self.args
        B, T,C = embds.size()
        if states is None:
            states = [None for _ in range(args.n_layer * 3)]
            for i in range(args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                states[i*3+0] = torch.zeros((B, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
                states[i*3+1] = torch.zeros((B, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
                states[i*3+2] = torch.zeros((B, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=embds.device)
        else:
            assert attention_mask.shape == (B, T), f'attention_mask shape: {attention_mask.shape}, embds shape: {embds.shape}'
        # if T%16 != 0:
        #     #right padding to 16x
        #     padding_length = 16 - T%16
        #     embds = torch.cat([torch.zeros(B, padding_length, C,device=embds.device,dtype=embds.dtype), embds], dim=1)
        #     attention_mask = torch.cat([torch.zeros(B, padding_length, dtype=torch.bool, device=embds.device), attention_mask], dim=1)
        attention_mask = attention_mask.unsqueeze(-1)
        x = embds
        v_first = torch.empty_like(x)
        for i in range(args.n_layer):
            block = self.blocks[i]
            tx_prev = states[i*3+0]
            state = states[i*3+1]
            cx_prev = states[i*3+2]
            x, v_first, tx_prev, state, cx_prev = block.forward_batch(x,attention_mask, v_first,tx_prev,state,cx_prev)
            states[i*3+0] = tx_prev
            states[i*3+1] = state
            states[i*3+2] = cx_prev
        x = x[:,-1,:]
        x = self.ln_out(x)
        logits = self.head(x)
        return x, logits, states
    
    def forward(self, embds,attention_mask = None):
        args = self.args
        B, T,C = embds.size()

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.bool, device=embds.device)
        else:
            assert attention_mask.shape == (B, T), f'attention_mask shape: {attention_mask.shape}, embds shape: {embds.shape}'
        if T%16 != 0:
            #right padding to 16x
            padding_length = 16 - T%16
            embds = torch.cat([torch.zeros(B, padding_length, C,device=embds.device,dtype=embds.dtype), embds], dim=1)
            attention_mask = torch.cat([torch.zeros(B, padding_length, dtype=torch.bool, device=embds.device), attention_mask], dim=1)
        attention_mask = attention_mask.unsqueeze(-1)
        x = embds
        if args.dropout > 0:
            x = self.drop0(x)
        v_first = torch.empty_like(x)
        for i in range(args.n_layer):
            block = self.blocks[i]
            if self.training and args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, attention_mask, v_first)
            else:
                x, v_first = block(x,attention_mask, v_first)
        x = self.ln_out(x)
        x = self.head(x)
        x = x[:, -T:]
        return x

    # def training_step(self, batch, batch_idx):
    #     args = self.args
        
    #     idx, targets,attention_mask = batch
    #     logits = self(idx,attention_mask)
    #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    #     return L2Wrap.apply(loss, logits)



class RWKV7ASRModelCuda(nn.Module):
    def __init__(self, whisper_encoder: WhisperEncoder, 
                audio_lm_model: RWKV7ModelForLatentInputsCuda,
                llm: RWKV7ModelForCausalLMCuda, 
                whisper_feature_extractor: WhisperFeatureExtractor):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.whisper_feature_extractor = whisper_feature_extractor
        self.projector1 = nn.Linear(whisper_encoder.config.hidden_size, audio_lm_model.args.n_embd)
        self.audio_lm_model = audio_lm_model
        self.projector2 = nn.Linear(audio_lm_model.args.n_embd, llm.args.n_embd)
        self.llm = llm
    def sample_logits(self, logits, top_k, top_p, temperature):
        """
        对logits进行采样，支持top-k和top-p采样
        输入: logits [batch_size, vocab_size]
        输出: sampled_tokens [batch_size, 1]
        """
        # 1. 应用温度缩放
        probs = F.softmax(logits.float() / temperature, dim=-1)
        
        # 2. 应用top-p采样 (nucleus sampling)
        if top_p < 1.0:
            # 对每个batch分别进行排序
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 找到累积概率超过top_p的索引
            # 使用torch.searchsorted的向量化版本
            top_p_tensor = torch.full((probs.shape[0],), top_p, device=probs.device, dtype=probs.dtype)
            cutoff_indices = torch.searchsorted(cumulative_probs, top_p_tensor.unsqueeze(-1), right=False)
            cutoff_indices = cutoff_indices.squeeze(-1)
            
            # 确保索引不超出范围
            cutoff_indices = torch.clamp(cutoff_indices, 0, sorted_probs.shape[-1] - 1)
            
            # 创建mask来过滤低概率的token
            batch_indices = torch.arange(probs.shape[0], device=probs.device)
            cutoff_probs = sorted_probs[batch_indices, cutoff_indices]
            
            # 将低于cutoff的概率设为0
            probs = torch.where(probs < cutoff_probs.unsqueeze(-1), torch.zeros_like(probs), probs)
        
        # 3. 应用top-k采样
        if top_k > 0:
            # 找到每个batch的top-k概率
            top_k_probs, _ = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)
            # 获取第k大的概率值
            kth_probs = top_k_probs[:, -1].unsqueeze(-1)
            # 将低于第k大概率的token概率设为0
            probs = torch.where(probs < kth_probs, torch.zeros_like(probs), probs)
        
        # 4. 重新归一化概率
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 5. 从分布中采样
        sampled_tokens = torch.multinomial(probs, num_samples=1)
        
        return sampled_tokens
    @torch.inference_mode()
    def forward_inference(self, audio_data, text_input_ids, text_attention_mask,hints_ids):
        batch_size = len(audio_data)
        # 1. 使用 whisper_feature_extractor 处理原始音频
        list_of_audio = []
        for audio in audio_data:
            if len(audio.shape) == 2:
                list_of_audio.append(audio.squeeze(0))
            else:
                list_of_audio.append(audio)
        
        features = self.whisper_feature_extractor(list_of_audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding_value=0.0)
        audio_attention_mask = features['attention_mask']
        
        # 确保张量在正确的设备上
        device = next(self.whisper_encoder.parameters()).device
        input_features = features['input_features'].to(dtype=torch.bfloat16).to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        
        # 2. 通过 whisper_encoder 编码音频特征
        with torch.no_grad():
            encoder_outputs = self.whisper_encoder(input_features)

        
        
        audio_latents = encoder_outputs.last_hidden_state  # [B, T_audio, hidden_size]
        if audio_attention_mask.shape[1] != audio_latents.shape[1]:
            # 计算下采样比例
            downsample_ratio = audio_attention_mask.shape[1] / audio_latents.shape[1]
            if not hasattr(self, 'downsample_printed'):
                print(f"Whisper下采样比例: {downsample_ratio:.2f} ({audio_attention_mask.shape[1]} -> {audio_latents.shape[1]})")
                self.downsample_printed = True
        else:
            downsample_ratio = 1.0
        projected_latents = self.projector1(audio_latents)  # [B, T_audio, hidden_size_of_llm]
        projected_latents = self.audio_lm_model.forward_batch(projected_latents)  # [B, T_audio, hidden_size]
        projected_latents = self.projector2(projected_latents)  # [B, T_audio, hidden_size_of_llm]

        # 3. 生成文本嵌入
        text_input_embeds = self.llm.emb(text_input_ids)  # [B, T_text, hidden_size_of_llm]
        
        # 4. 处理hints_ids：如果是一维的，扩展成 (B, T_hints)
        if hints_ids is not None:
            if hints_ids.dim() == 1:
                hints_ids = hints_ids.unsqueeze(0).expand(batch_size, -1)
            hints_embeds = self.llm.emb(hints_ids)  # [B, T_hints, hidden_size_of_llm]
        else:
            hints_embeds = None
        # 6. 计算下采样比例，处理attention mask不匹配问题
        # WhisperEncoder内部有下采样，将原始特征压缩到更少的帧
        
        
        # 7. 遍历所有样本，根据attention mask连接有效部分
        valid_embeds_list = []
        valid_attention_mask_list = []
        
        # 添加调试信息和错误处理
        if not hasattr(self, 'debug_printed'):
            print(f"Debug: audio_attention_mask shape: {audio_attention_mask.shape}")
            print(f"Debug: audio_attention_mask dtype: {audio_attention_mask.dtype}")
            print(f"Debug: audio_attention_mask device: {audio_attention_mask.device}")
            print(f"Debug: audio_latents shape: {audio_latents.shape}")
            print(f"Debug: downsample_ratio: {downsample_ratio}")
            self.debug_printed = True
        
        # 确保数据类型正确
        if audio_attention_mask.dtype != torch.long:
            audio_attention_mask = audio_attention_mask.long()
        if text_attention_mask.dtype != torch.long:
            text_attention_mask = text_attention_mask.long()

        # 添加安全的索引检查
        try:
            # 使用原始attention mask计算有效长度，然后除以下采样比例
            audio_valid_lengths = audio_attention_mask.sum(dim=1)
            text_valid_lengths = text_attention_mask.sum(dim=1)
        except Exception as e:
            print(f"Error in attention mask sum: {e}")
            print(f"audio_attention_mask shape: {audio_attention_mask.shape}")
            print(f"text_attention_mask shape: {text_attention_mask.shape}")
            raise e
        
        for i in range(batch_size):
            # 获取当前样本的有效长度
            # 关键修复：将原始attention mask计算的长度除以下采样比例
            audio_valid_length = int(audio_valid_lengths[i].item() / downsample_ratio)+1
            text_valid_length = text_valid_lengths[i].item()
            
            # 获取有效的音频嵌入（右padding，有效元素在左边）
            audio_valid_embeds = projected_latents[i, :audio_valid_length] if audio_valid_length > 0 else torch.empty(0, projected_latents.size(-1), device=projected_latents.device, dtype=projected_latents.dtype)
            
            # 获取有效的文本嵌入（左padding，有效元素在右边）
            text_valid_embeds = text_input_embeds[i, -text_valid_length:] if text_valid_length > 0 else torch.empty(0, text_input_embeds.size(-1), device=text_input_embeds.device, dtype=text_input_embeds.dtype)
            
            # 获取hints嵌入
            hints_valid_embeds = None
            if hints_embeds is not None:
                hints_valid_embeds = hints_embeds[i]  # [T_hints, hidden_size]
            
            # 按照顺序连接：text_embeds + audio_embeds + hints_embeds + labels_embeds
            embed_parts = [text_valid_embeds, audio_valid_embeds]
            if hints_valid_embeds is not None:
                embed_parts.append(hints_valid_embeds)
            
            combined_embeds = torch.cat(embed_parts, dim=0)  # [T_total, hidden_size]
            valid_embeds_list.append(combined_embeds)
            
            # 生成全1的attention mask
            valid_attention_mask = torch.ones(len(combined_embeds), dtype=torch.long, device=audio_attention_mask.device)
            valid_attention_mask_list.append(valid_attention_mask)
            
        
        # 7. 使用pad_sequence进行左对齐
        input_embeds = pad_sequence(valid_embeds_list, batch_first=True, padding_value=0.0, padding_side='left')
        attention_mask = pad_sequence(valid_attention_mask_list, batch_first=True, padding_value=0, padding_side='left')
        
        # 调试信息（只在第一个batch打印）
        if not hasattr(self, "first_batch"):
            print(f'input_embeds shape: {input_embeds.shape}')
            print(f'attention_mask shape: {attention_mask.shape}')
            self.first_batch = True

        x,logits,states = self.llm.forward_batch(input_embeds,attention_mask)
        result_tokens = [[] for i in range(batch_size)]
        next_tokens = self.sample_logits(logits,top_k=1,top_p=1,temperature=1.0)
        for i in range(batch_size):
            result_tokens[i].append(next_tokens[i].item())
        print(f'next_tokens: {next_tokens}')
        max_word = 10
        while True:
            embeds = self.llm.emb(next_tokens)
            x,logits,states = self.llm.forward_batch(embeds,states=states)
            print(f'logits: {logits}')
            next_tokens = self.sample_logits(logits,top_k=1,top_p=1,temperature=1.0)
            finished_batches = 0
            for i in range(batch_size):
                if result_tokens[i][-1] == 0:
                    result_tokens[i].append(0)
                else:
                    result_tokens[i].append(next_tokens[i].item())
                print(f'result_tokens[{i}]: {result_tokens[i]}')
                if result_tokens[i][-1] == 0:
                    finished_batches += 1
            if finished_batches == batch_size:
                break
        return result_tokens

            
    def forward(self, audio_data, text_input_ids, text_attention_mask, labels=None, labels_attention_mask=None, hints_ids=None):
        """
        重新设计的forward方法，按照正确的逻辑处理数据，参照rwkv_asr.py的格式
        
        Args:
            audio_data: 原始音频数据列表
            text_input_ids: 左对齐的指令文本tokens [B, T_text]
            text_attention_mask: 文本attention mask [B, T_text]
            labels: 目标标签 [B, T_labels]
            labels_attention_mask: 标签attention mask [B, T_labels]
            hints_ids: 提示词tokens [T_hints] 或 [B, T_hints]
        """
        batch_size = len(audio_data)
        
        # 1. 使用 whisper_feature_extractor 处理原始音频
        list_of_audio = []
        for audio in audio_data:
            if len(audio.shape) == 2:
                list_of_audio.append(audio.squeeze(0))
            else:
                list_of_audio.append(audio)
        
        features = self.whisper_feature_extractor(list_of_audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding_value=0.0)
        input_features = features['input_features']
        audio_attention_mask = features['attention_mask']
        # audio_attention_mask = audio_attention_mask.squeeze(0)
        
        # 确保张量在正确的设备上
        device = next(self.whisper_encoder.parameters()).device
        input_features = input_features.to(dtype=torch.bfloat16).to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        # 2. 通过 whisper_encoder 编码音频特征
        with torch.no_grad():
            encoder_outputs = self.whisper_encoder(input_features)
        audio_latents = encoder_outputs.last_hidden_state  # [B, T_audio, hidden_size]
        projected_latents = self.projector1(audio_latents)  # [B, T_audio, hidden_size_of_llm]
        projected_latents = self.audio_lm_model(projected_latents)  # [B, T_audio, hidden_size]
        projected_latents = self.projector2(projected_latents)  # [B, T_audio, hidden_size_of_llm]
        
        # 3. 生成文本嵌入
        text_input_embeds = self.llm.emb(text_input_ids)  # [B, T_text, hidden_size_of_llm]
        
        # 4. 处理hints_ids：如果是一维的，扩展成 (B, T_hints)
        if hints_ids is not None:
            if hints_ids.dim() == 1:
                hints_ids = hints_ids.unsqueeze(0).expand(batch_size, -1)
            hints_embeds = self.llm.emb(hints_ids)  # [B, T_hints, hidden_size_of_llm]
        else:
            hints_embeds = None
        
        # 5. 生成标签嵌入（如果提供）- 关键修复：处理-100值
        if labels is not None and labels_attention_mask is not None:
            cloned_labels = labels.clone()
            # 将-100设置为0，避免embedding溢出
            cloned_labels[cloned_labels == -100] = 0
            labels_embeds = self.llm.emb(cloned_labels)  # [B, T_labels, hidden_size_of_llm]
        else:
            labels_embeds = None
        
        # 6. 计算下采样比例，处理attention mask不匹配问题
        # WhisperEncoder内部有下采样，将原始特征压缩到更少的帧
        if audio_attention_mask.shape[1] != audio_latents.shape[1]:
            # 计算下采样比例
            downsample_ratio = audio_attention_mask.shape[1] / audio_latents.shape[1]
            if not hasattr(self, 'downsample_printed'):
                print(f"Whisper下采样比例: {downsample_ratio:.2f} ({audio_attention_mask.shape[1]} -> {audio_latents.shape[1]})")
                self.downsample_printed = True
        else:
            downsample_ratio = 1.0
        
        # 7. 遍历所有样本，根据attention mask连接有效部分
        valid_embeds_list = []
        valid_attention_mask_list = []
        valid_labels_list = []
        
        # 添加调试信息和错误处理
        if not hasattr(self, 'debug_printed'):
            print(f"Debug: audio_attention_mask shape: {audio_attention_mask.shape}")
            print(f"Debug: audio_attention_mask dtype: {audio_attention_mask.dtype}")
            print(f"Debug: audio_attention_mask device: {audio_attention_mask.device}")
            print(f"Debug: audio_latents shape: {audio_latents.shape}")
            print(f"Debug: downsample_ratio: {downsample_ratio}")
            self.debug_printed = True
        
        # 确保数据类型正确
        if audio_attention_mask.dtype != torch.long:
            audio_attention_mask = audio_attention_mask.long()
        if text_attention_mask.dtype != torch.long:
            text_attention_mask = text_attention_mask.long()
        if labels_attention_mask.dtype != torch.long:
            labels_attention_mask = labels_attention_mask.long()
        
        # 添加安全的索引检查
        try:
            # 使用原始attention mask计算有效长度，然后除以下采样比例
            audio_valid_lengths = audio_attention_mask.sum(dim=1)
            text_valid_lengths = text_attention_mask.sum(dim=1)
            labels_valid_lengths = labels_attention_mask.sum(dim=1) if labels_attention_mask is not None else torch.zeros(batch_size, dtype=torch.long, device=device)
        except Exception as e:
            print(f"Error in attention mask sum: {e}")
            print(f"audio_attention_mask shape: {audio_attention_mask.shape}")
            print(f"text_attention_mask shape: {text_attention_mask.shape}")
            print(f"labels_attention_mask shape: {labels_attention_mask.shape if labels_attention_mask is not None else 'None'}")
            raise e
        
        for i in range(batch_size):
            # 获取当前样本的有效长度
            # 关键修复：将原始attention mask计算的长度除以下采样比例
            audio_valid_length = int(audio_valid_lengths[i].item() / downsample_ratio)+1
            text_valid_length = text_valid_lengths[i].item()
            labels_valid_length = labels_valid_lengths[i].item() if labels_attention_mask is not None else 0
            
            # 获取有效的音频嵌入（右padding，有效元素在左边）
            audio_valid_embeds = projected_latents[i, :audio_valid_length] if audio_valid_length > 0 else torch.empty(0, projected_latents.size(-1), device=projected_latents.device, dtype=projected_latents.dtype)
            
            # 获取有效的文本嵌入（左padding，有效元素在右边）
            text_valid_embeds = text_input_embeds[i, -text_valid_length:] if text_valid_length > 0 else torch.empty(0, text_input_embeds.size(-1), device=text_input_embeds.device, dtype=text_input_embeds.dtype)
            
            # 获取hints嵌入
            hints_valid_embeds = None
            if hints_embeds is not None:
                hints_valid_embeds = hints_embeds[i]  # [T_hints, hidden_size]
            
            # 获取标签嵌入
            labels_valid_embeds = None
            if labels_embeds is not None and labels_attention_mask is not None:
                labels_valid_length = labels_attention_mask[i].sum().item()
                labels_valid_embeds = labels_embeds[i, -labels_valid_length:] if labels_valid_length > 0 else torch.empty(0, labels_embeds.size(-1), device=labels_embeds.device, dtype=labels_embeds.dtype)
            
            # 按照顺序连接：text_embeds + audio_embeds + hints_embeds + labels_embeds
            embed_parts = [text_valid_embeds, audio_valid_embeds]
            if hints_valid_embeds is not None:
                embed_parts.append(hints_valid_embeds)
            if labels_valid_embeds is not None:
                embed_parts.append(labels_valid_embeds)
            
            combined_embeds = torch.cat(embed_parts, dim=0)  # [T_total, hidden_size]
            valid_embeds_list.append(combined_embeds)
            
            # 生成全1的attention mask
            valid_attention_mask = torch.ones(len(combined_embeds), dtype=torch.long, device=audio_attention_mask.device)
            valid_attention_mask_list.append(valid_attention_mask)
            
            # 生成labels：只对labels部分计算损失，其他部分设为-100
            if labels is not None and labels_attention_mask is not None:
                # 创建全-100的tensor
                sample_labels = torch.full((len(combined_embeds),), -100, dtype=labels.dtype, device=labels.device)
                
                # 只对labels部分赋值（由于左对齐padding，labels总是在最右边）
                if len(labels_valid_embeds) > 0:
                    labels_len = len(labels_valid_embeds)
                    sample_labels[-labels_len:] = labels[i, -labels_len:]
                
                valid_labels_list.append(sample_labels)
            else:
                # 如果没有labels，创建全-100的tensor
                sample_labels = torch.full((len(combined_embeds),), -100, dtype=torch.long, device=audio_attention_mask.device)
                valid_labels_list.append(sample_labels)
        
        # 7. 使用pad_sequence进行左对齐
        input_embeds = pad_sequence(valid_embeds_list, batch_first=True, padding_value=0.0, padding_side='left')
        attention_mask = pad_sequence(valid_attention_mask_list, batch_first=True, padding_value=0, padding_side='left')
        final_labels = pad_sequence(valid_labels_list, batch_first=True, padding_value=-100, padding_side='left')
        
        # 调试信息（只在第一个batch打印）
        if not hasattr(self, "first_batch"):
            print(f'input_embeds shape: {input_embeds.shape}')
            print(f'attention_mask shape: {attention_mask.shape}')
            print(f'labels shape: {final_labels.shape}')
            print(f'labels sample: {final_labels}')  
            self.first_batch = True
        
        # 8. 调用LLM
        if labels is not None:
            # 对于CUDA版本，我们需要手动计算损失
            logits = self.llm(input_embeds, attention_mask=attention_mask)
            # 计算交叉熵损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = final_labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            loss = L2Wrap.apply(loss, logits)
            return type('Output', (), {'loss': loss, 'logits': logits})()
        else:
            logits = self.llm(input_embeds, attention_mask=attention_mask)
            return type('Output', (), {'logits': logits})()



def load_whisper_feature_extractor_and_encoder(whisper_path):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
    print(f"Loaded WhisperFeatureExtractor: {feature_extractor}")
    whisper_config = WhisperConfig.from_pretrained(whisper_path)
    print(f"Loaded WhisperConfig: {whisper_config}")
    encoder = WhisperEncoder(whisper_config)
    print(f"Created WhisperEncoder: {encoder}")
    
    # 加载预训练权重
    full_model_state_dict = torch.load(os.path.join(whisper_path, "pytorch_model.bin"),map_location=torch.device("cpu"))
    encoder_state_dict = OrderedDict()
    encoder_prefix = "model.encoder."
    
    # 检查是否已经有encoder前缀的键
    has_encoder_prefix = any(key.startswith(encoder_prefix) for key in full_model_state_dict.keys())
    
    if has_encoder_prefix:
        # 如果有前缀，需要去掉前缀
        for key, value in full_model_state_dict.items():
            if key.startswith(encoder_prefix):
                new_key = key[len(encoder_prefix):]
                encoder_state_dict[new_key] = value
    else:
        # 如果没有前缀，直接使用所有键
        encoder_state_dict = full_model_state_dict
    
    encoder.load_state_dict(encoder_state_dict)
    print(f"Loaded encoder weights from {whisper_path}")
    
    return feature_extractor, encoder


