from types import SimpleNamespace
import torch
import torch.nn as nn
from src.utils.utils import ModelConfig
from src.models.layers import CausalSelfAttention, FFN


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


class TransformerBackBone(nn.Module):
    """
    A transformer backbone model similar to GPT-2. Returns last hidden states.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        H = config.n_embd
        S = config.n_positions

        # positional embeddings
        self.wpe = nn.Embedding(S, H)

        # input dropout
        self.dropout = nn.Dropout(config.embd_pdrop)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(H, eps=1e-5)

        # init weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        output_hidden_states=False,
    ):
        """
        inputs_embeds: [B, S, H] (preferred for your use-case)
        attention_mask: optional [1, 1, S, S] boolean mask (True = keep, False = mask out)
        output_hidden_states: whether to return all hidden states
        """
        B, S, H = inputs_embeds.shape
        if S > self.config.n_positions:
            raise ValueError(f"Sequence length {S} > n_positions {self.config.n_positions}")

        # add positional embeddings
        pos_ids = torch.arange(S, device=inputs_embeds.device).unsqueeze(0)  # [1, S]
        pos_emb = self.wpe(pos_ids)  # [1, S, H]
        hidden = inputs_embeds + pos_emb
        hidden = self.dropout(hidden)

        hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            if output_hidden_states:
                hidden_states = (*hidden_states, hidden)
            hidden = block(hidden, attention_mask=attention_mask)

        hidden = self.ln_f(hidden)

        if output_hidden_states:
            hidden_states = (*hidden_states, hidden)

        # return a simple namespace compatible object with `last_hidden_state`
        out = SimpleNamespace(last_hidden_state=hidden, hidden_states=hidden_states)
        return out



if __name__ == "__main__":
    cfg = ModelConfig(
        n_positions=64,
        n_embd=128,
        n_layer=4,
        n_head=8,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = TransformerBackBone(cfg)
    B, S = 2, 10
    x = torch.randn(B, S, cfg.n_embd)  # inputs_embeds
    out = model(inputs_embeds=x, output_hidden_states=True)
    print(out.last_hidden_state.shape)  # -> [2, 10, 128]
    for h in out.hidden_states:
        print(h.shape)  # -> [2, 10, 128]
