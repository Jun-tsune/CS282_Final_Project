import math
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    """A minimal config object with same fields as your GPT2Config usage."""
    def __init__(
        self,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        use_cache=False,
    ):
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.use_cache = use_cache


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
        # x: [B, T, C]
        B, T, C = x.size()
        qkv = self.qkv(x)  # [B, T, 3*C]
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)  # [B, T, 3, H, Dh]
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, Dh]

        # transpose to [B, H, T, Dh]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # attention scores [B, H, T, T]
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # causal mask: allow positions j <= i
        # make mask shape [1, 1, T, T]
        causal_mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # optional user attention_mask: expects shape broadcastable to [B, 1, 1, T] or [B, 1, T, T]
        if attention_mask is not None:
            # support 1D attention mask of shape [B, T] where 1 means keep, 0 means mask
            if attention_mask.dim() == 2:
                # make it [B, 1, 1, T]
                am = attention_mask[:, None, None, :].to(torch.bool)
                att = att.masked_fill(~am, float('-inf'))
            elif attention_mask.dim() == 4:
                # assume already [B, 1, T, T] or [B, H, T, T]
                att = att.masked_fill(~attention_mask, float('-inf'))
            else:
                raise ValueError("Unsupported attention_mask shape")

        att_weights = F.softmax(att, dim=-1)  # [B, H, T, T]
        att_weights = self.attn_dropout(att_weights)
        out = torch.matmul(att_weights, v)  # [B, H, T, Dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)  # [B, T, C]
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


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.ffn = FFN(config)

    def forward(self, x, attention_mask=None):
        # pre-norm
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln_2(x))
        return x


class SimpleGPT2Like(nn.Module):
    """
    A lightweight GPT-2 like model.
    Interface:
        model = SimpleGPT2Like(config)
        out = model(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
        last_states = out.last_hidden_state  # [B, T, C]
    It will also accept input_ids if you add a token embedding externally.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        C = config.n_embd
        T = config.n_positions

        # token embedding not included by default (we expect inputs_embeds),
        # but provide a small token embedding option if user wants to use.
        self.wte = nn.Embedding(1, C)  # dummy; you can replace/ignore

        # positional embeddings
        self.wpe = nn.Embedding(T, C)

        # input dropout
        self.dropout = nn.Dropout(config.embd_pdrop)

        # stack of transformer blocks
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(C, eps=1e-5)

        # init weights
        self._init_weights()

    def _init_weights(self):
        # simple initialization similar to HF
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        use_cache=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        inputs_embeds: [B, T, C] (preferred for your use-case)
        attention_mask: optional [B, T] (1 means keep, 0 means mask)
        """
        if use_cache is None:
            use_cache = self.config.use_cache

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must provide input_ids or inputs_embeds")
            inputs_embeds = self.wte(input_ids)  # if you actually use token ids

        B, T, C = inputs_embeds.shape
        if T > self.config.n_positions:
            raise ValueError(f"Sequence length {T} > n_positions {self.config.n_positions}")

        # add positional embeddings
        pos_ids = torch.arange(T, device=inputs_embeds.device).unsqueeze(0)  # [1, T]
        pos_emb = self.wpe(pos_ids)  # [1, T, C]
        hidden = inputs_embeds + pos_emb
        hidden = self.dropout(hidden)

        hidden_states = () if output_hidden_states else None
        for block in self.h:
            if output_hidden_states:
                hidden_states = (*hidden_states, hidden)
            hidden = block(hidden, attention_mask=attention_mask)

        hidden = self.ln_f(hidden)

        if output_hidden_states:
            hidden_states = (*hidden_states, hidden)

        # return a simple namespace compatible object with `last_hidden_state`
        out = SimpleNamespace(last_hidden_state=hidden, hidden_states=hidden_states)
        return out


# Example usage:
if __name__ == "__main__":
    cfg = ModelConfig(
        n_positions=64,
        n_embd=128,
        n_layer=4,
        n_head=8,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        use_cache=False,
    )
    model = SimpleGPT2Like(cfg)
    B, T = 2, 10
    x = torch.randn(B, T, cfg.n_embd)  # inputs_embeds
    out = model(inputs_embeds=x)  # returns object with out.last_hidden_state
    print(out.last_hidden_state.shape)  # -> [2, 10, 128]
