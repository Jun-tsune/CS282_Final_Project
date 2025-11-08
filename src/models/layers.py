import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utils_model import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
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
    def __init__(self, config: ModelConfig):
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