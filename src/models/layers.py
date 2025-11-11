import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

Memory = namedtuple("Memory", ["mem", "compressed_mem"])

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # qkv projection and out projection
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # causal mask will be created on the fly in forward based on seq_len

    def forward(self, x, attention_mask=None):
        # x: [B, S, H]
        # B: batch size
        # S: sequence length
        # H: embedding dim
        # Nh, Dh: number of heads, head dim
        # attention_mask: Optional [1, 1, S, S], boolean mask (True = keep, False = mask out)
        B, S, H = x.size()
        qkv = self.qkv(x)  # [B, S, 3*H]
        qkv = qkv.view(B, S, 3, self.n_head, self.head_dim)  # [B, S, 3, Nh, Dh]
        q, k, v = qkv.unbind(dim=2)  # each [B, S, Nh, Dh]

        # transpose to [B, Nh, S, Dh]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attention scores [B, Nh, S, S]
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # optional user attention_mask
        if attention_mask is not None:
            # Check the shape of attention_mask
            if attention_mask.dim() == 4 and attention_mask.shape[0] == 1 and attention_mask.shape[1] == 1 and attention_mask.shape[2] == S and attention_mask.shape[3] == S:
                attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
                att = att.masked_fill(~attention_mask, float('-inf'))
            else:
                raise ValueError("Unsupported attention_mask shape. Expected 4D tensor (1, 1, S, S).")
        else:
            # default causal mask
            attention_mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
            att = att.masked_fill(~attention_mask, float('-inf'))

        att_weights = F.softmax(att, dim=-1)  # [B, Nh, S, S]
        att_weights = self.attn_dropout(att_weights)
        out = torch.matmul(att_weights, v)  # [B, Nh, S, Dh]
        out = out.transpose(1, 2).contiguous().view(B, S, H)  # [B, S, H]
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, inner_dim)
        self.fc2 = nn.Linear(inner_dim, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Compressive Transformer Attention


# helper functions

def split_at_index(dim, index, t):
    pre = (slice(None),) * dim
    l = (*pre, slice(None, index))
    r = (*pre, slice(index, None))
    return t[l], t[r]

def queue_fifo(*args, length, dim=-2):
    q = torch.cat(args, dim=dim)
    if length > 0:
        return split_at_index(dim, -length, q)
    device = q.device
    shape = list(q.shape); shape[dim] = 0
    return q, torch.empty(shape, device=device, dtype=q.dtype)

def shift_pos_logits(x):
    # shift relative position
    *_, i, j = x.shape
    device, dtype = x.device, x.dtype
    zero_pad = torch.zeros((*_, i, i), device=device, dtype=dtype)
    x = torch.cat([x, zero_pad], -1)
    L = i + j - 1
    x = x.view(*_, -1)
    zero_pad2 = torch.zeros(*_, -x.size(-1) % L, device=device, dtype=dtype)
    shifted = torch.cat([x, zero_pad2], -1).view(*_, -1, L)
    return shifted[..., :i, i - 1:]

class ConvCompress(nn.Module):
    """Use 1D Conv(stride=ratio) Compress memory """
    def __init__(self, dim, ratio: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=ratio, stride=ratio)
    def forward(self, mem):               # [B, Tm, H]
        mem = mem.transpose(1, 2)        # [B, H, Tm]
        out = self.conv(mem)             # [B, H, Tm/ratio]
        return out.transpose(1, 2)       # [B, Tm/ratio, H]


class CompressCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q comes from x；K/V come from [cmem, mem, x]
        self.to_q  = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.to_kv = nn.Linear(config.n_embd, 2 * config.n_embd, bias=False)
        self.proj  = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout  = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # compress parameters
        self.seq_len    = getattr(config, "n_positions")
        self.mem_len    = getattr(config, "mem_len", self.seq_len)  # default=seq_len
        self.cmem_ratio = max(1, getattr(config, "cmem_ratio", 4))
        self.cmem_len   = getattr(config, "cmem_len", self.mem_len // self.cmem_ratio)

        self.compress = ConvCompress(config.n_embd, self.cmem_ratio)
        self.recon_attn_dropout = nn.Dropout(getattr(config, "recon_attn_dropout", 0.0))

    # heads reshape
    def split_heads(self, x):  # [B, T, H] -> [B, Nh, T, Dh]
        B, T, H = x.size()
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    @torch.no_grad()
    def _build_causal_mask(self, S, total_mem_len, device, dtype=torch.bool):
        # [1,1,S, S+total_mem_len]
        # Use Triangle mask
        # only see memory and past tokens, can't see future tokens
        M = S + total_mem_len
        tri = torch.triu(torch.ones(S, M, device=device, dtype=dtype), diagonal=1+total_mem_len)
        return (~tri).view(1, 1, S, M)


    def forward(
        self,
        x,                         # [B, S, H]
        attention_mask=None,       # [1,1,S,S] True=keep
        memories=None,             # (mem, cmem)
        pos_bias=None,             # [Nh, M, Dh] (relative position bias, This is Important!)
        calc_memory=True,
    ):
        B, S, H = x.shape
        device, dtype = x.device, x.dtype

        # unpack memories
        mem, cmem = (None, None) if memories is None else memories
        if mem  is None: mem  = torch.empty(B, 0, H, device=device, dtype=dtype)
        if cmem is None: cmem = torch.empty(B, 0, H, device=device, dtype=dtype)
        mem_len, cmem_len = mem.size(1), cmem.size(1)
        total_mem_len = mem_len + cmem_len

        # attn calculation
        q = self.to_q(x)                              # [B,S,H]
        kv_in = torch.cat([cmem, mem, x], dim=1)     # [B,M,H], M = cmem + mem + S
        k, v = self.to_kv(kv_in).chunk(2, dim=-1)    # chunk into two parts each [B,M,H]

        q = self.split_heads(q)   # [B,Nh,S,Dh]
        k = self.split_heads(k)   # [B,Nh,M,Dh]
        v = self.split_heads(v)   # [B,Nh,M,Dh]

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,Nh,S,M]
        
        # use smallest neg value to mask
        neg = -torch.finfo(att.dtype).max

        # relative positison shift
        # put mem and cmem far away
        if pos_bias is not None:
            # pos_bias [Nh, M, Dh]
            M = k.size(-2)
            pos = pos_bias[:, -M:].to(dtype=q.dtype, device=q.device)  # [Nh,M,Dh]
            pos_logits = torch.einsum("bhsd,hmd->bhsm", q, pos) * self.scale
            pos_logits = shift_pos_logits(pos_logits)
            att = att + pos_logits

        if attention_mask is None:
            causal = self._build_causal_mask(S, total_mem_len, device=device)  # True=keep
            att = att.masked_fill(~causal, neg)
        else:
            if attention_mask.dim()==4 and attention_mask.shape[:2]==(1,1) and attention_mask.shape[2]==S and attention_mask.shape[3]==S:
                user = attention_mask.to(device=device, dtype=torch.bool)  # True=keep
                user = F.pad(user, (total_mem_len, 0), value=True)         # [1,1,S,M]
                causal = self._build_causal_mask(S, total_mem_len, device=device)
                mask = user & causal
                att = att.masked_fill(~mask, neg)
            else:
                raise ValueError("Unsupported attention_mask shape. Expected [1,1,S,S].")

        att_w = F.softmax(att, dim=-1)
        att_w = self.attn_dropout(att_w)
        out = torch.matmul(att_w, v)  # [B,Nh,S,Dh]
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        out = self.proj(out)
        out = self.resid_dropout(out)

        # memory update
        new_mem, new_cmem = mem, cmem
        aux_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)

        # edge case
        if self.seq_len and (S < self.seq_len or not calc_memory):
            return out, Memory(new_mem, new_cmem), aux_loss

        # FIFO queue: generate new mem；old mem go to cmem
        old_mem, new_mem = queue_fifo(mem, x, length=self.mem_len, dim=1)
        pad = old_mem.size(1) % self.cmem_ratio
        if pad != 0:
            old_mem = F.pad(old_mem, (0, 0, pad, 0), value=0.0)

        if old_mem.size(1)==0 or self.cmem_len<=0: # cap is 0
            return out, Memory(new_mem, new_cmem), aux_loss

        compressed = self.compress(old_mem.detach())         # [B,Told/ratio,H]
        concat_c = torch.cat([cmem, compressed], dim=1)      # [B, cmem + compressed, H]
        _, new_cmem = split_at_index(1, -self.cmem_len, concat_c) # if queue len > cmem_len drop old mem

        # Reconstruct Loss (only train needs)
        # In order to learn compress efficiently
        if self.training:
            self.to_kv.weight.detach_()

            cmem_k, cmem_v = self.to_kv(compressed).chunk(2, dim=-1)
            cmem_k = self.split_heads(cmem_k)  # [B,Nh,Tc,Dh]
            cmem_v = self.split_heads(cmem_v)

            k_all, v_all = k, v                   # without compressive [B,Nh,M,Dh]
            take = min(mem_len, self.mem_len)
            old_rng = slice(-(take + S), -S) if take>0 else slice(0, 0)
            old_mem_k = k_all[:, :, old_rng].detach().clone()
            old_mem_v = v_all[:, :, old_rng].detach().clone()
            q_det = q.detach()

            def attn_full(q, k, v, drop):
                dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn = F.softmax(dots, dim=-1)
                if drop is not None: attn = drop(attn)
                return torch.matmul(attn, v)

            pred_old  = attn_full(q_det, old_mem_k, old_mem_v, self.recon_attn_dropout)
            pred_cmem = attn_full(q_det, cmem_k, cmem_v, self.recon_attn_dropout)
            aux_loss = F.mse_loss(pred_old, pred_cmem)

        return out, Memory(new_mem, new_cmem), aux_loss