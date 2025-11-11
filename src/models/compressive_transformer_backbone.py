from types import SimpleNamespace
import torch
import torch.nn as nn
from src.utils.utils import CompressModelConfig
from src.models.layers import CompressCausalSelfAttention, FFN
from collections import namedtuple

Memory = namedtuple("Memory", ["mem", "compressed_mem"])

class CompressTransformerBlock(nn.Module):
    def __init__(self, config: CompressModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CompressCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.ffn = FFN(config)

    def forward(self, x, attention_mask=None, memories=None, pos_bias=None, calc_memory=True):
        # pre-norm
        attn_out, memory_out, aux_loss = self.attn(self.ln_1(x),
                                                   attention_mask=attention_mask,
                                                   memories=memories,
                                                   pos_bias=pos_bias,
                                                   calc_memory=calc_memory)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, memory_out, aux_loss


class CompressTransformerBackBone(nn.Module):
    """
    A compressive transformer backbone model similar to GPT-2. 
    Returns last hidden states, Memory and aux_loss.
    """
    def __init__(self, config: CompressModelConfig):
        super().__init__()
        self.config = config
        H = config.n_embd
        S = config.n_positions
        self.depth = config.n_layer

        # Absolute pos embedding
        self.wpe = nn.Embedding(S, H)
        self.dropout = nn.Dropout(config.embd_pdrop)

        # which layers open memory（default all layers）
        self.memory_layers = list(range(1, self.depth + 1))
        assert all(1 <= l <= self.depth for l in self.memory_layers)

        # relative pos embedding
        seq_len = S
        mem_len = getattr(config, "mem_len", S)
        cmem_len = getattr(config, "cmem_len", max(1, mem_len // getattr(config, "cmem_ratio", 4)))
        self.register_parameter(
            "pos_bias",
            nn.Parameter(torch.zeros(config.n_head, seq_len + mem_len + cmem_len, H // config.n_head))
        )

        # Add Block
        self.blocks = nn.ModuleList([CompressTransformerBlock(config) for _ in range(self.depth)])
        self.ln_f = nn.LayerNorm(H, eps=1e-5)

        self.reconstruction_loss_weight = getattr(config, "reconstruction_loss_weight", 1.0)

        # init
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        inputs_embeds,                  # [B,S,H]
        attention_mask=None,            # [1,1,S,S]
        memories=None,                  # (mem, cmem) or None；shape: [Lmem,B,T,H]
        output_hidden_states=False,
        enhanced_recurrence=False       
    ):
        B, S, H = inputs_embeds.shape
        if S > self.config.n_positions:
            raise ValueError(f"Sequence length {S} > n_positions {self.config.n_positions}")

        # abs position embedding
        pos_ids = torch.arange(S, device=inputs_embeds.device).unsqueeze(0)  # [1,S]
        pos_emb = self.wpe(pos_ids)  # [1,S,H]
        hidden = self.dropout(inputs_embeds + pos_emb)

        # memories: (mem, cmem) with shape [Lmem, B, T, H]
        if memories is None:
            mem = torch.empty(len(self.memory_layers), B, 0, H, device=hidden.device, dtype=hidden.dtype)
            cmem = torch.empty(len(self.memory_layers), B, 0, H, device=hidden.device, dtype=hidden.dtype)
        else:
            mem, cmem = memories
            # shape check
            assert mem.dim()==4 and cmem.dim()==4 and mem.size(0)==len(self.memory_layers) and cmem.size(0)==len(self.memory_layers)

        # one trick to better results
        if enhanced_recurrence and len(self.memory_layers) > 0:
            mem  = torch.roll(mem, -1, dims=0)
            cmem = torch.roll(cmem, -1, dims=0)

        # mem + cmem + s
        total_len = mem.size(2) + cmem.size(2) + S
        pos_bias = self.pos_bias[:, -total_len:]  # [Nh, M, Dh]

        hidden_states = () if output_hidden_states else None
        next_mem, next_cmem = [], []
        aux_loss_total = torch.tensor(0., device=hidden.device, dtype=hidden.dtype, requires_grad=True)

        mem_iter = iter(mem)
        cmem_iter = iter(cmem)

        for li, block in enumerate(self.blocks, start=1):
            if output_hidden_states:
                hidden_states = (*hidden_states, hidden)

            use_memory = (li in self.memory_layers)
            layer_memories = None
            if use_memory:
                layer_memories = (next(mem_iter), next(cmem_iter))  # [B,T,H],[B,Tc,H]

            hidden, layer_memory_out, layer_aux = block(
                hidden,
                attention_mask=attention_mask,
                memories=layer_memories,
                pos_bias=pos_bias if use_memory else None,
                calc_memory=use_memory
            )

            if use_memory:
                next_mem.append(layer_memory_out.mem)               # [B,Tm_new,H]
                next_cmem.append(layer_memory_out.compressed_mem)   # [B,Tc_new,H]
                aux_loss_total = aux_loss_total + layer_aux

        hidden = self.ln_f(hidden)
        if output_hidden_states:
            hidden_states = (*hidden_states, hidden)

        # memories:[Lmem,B,T,H]
        if len(next_mem) == 0:
            next_mem_stack = torch.empty(0, B, 0, H, device=hidden.device, dtype=hidden.dtype)
            next_cmem_stack = torch.empty(0, B, 0, H, device=hidden.device, dtype=hidden.dtype)
        else:
            next_mem_stack  = torch.stack(next_mem, dim=0).detach()
            next_cmem_stack = torch.stack(next_cmem, dim=0).detach()

        # aux_loss_total
        num_memory_layers = max(1, len(self.memory_layers))
        aux_loss_total = aux_loss_total * (self.reconstruction_loss_weight / num_memory_layers)

        out = SimpleNamespace(last_hidden_state=hidden, hidden_states=hidden_states)
        return out, Memory(mem=next_mem_stack, compressed_mem=next_cmem_stack), aux_loss_total


# if __name__ == "__main__":
#     cfg = CompressModelConfig(
#         n_positions=64,
#         n_embd=128,
#         n_layer=4,
#         n_head=8,
#         resid_pdrop=0.1,
#         embd_pdrop=0.1,
#         attn_pdrop=0.1,
#         # compressive related
#         mem_len=64,
#         cmem_ratio=4,
#         # memory_layers=[1,3],
#         recon_attn_dropout=0.0,
#         reconstruction_loss_weight=1.0,
#     )
#     model = CompressTransformerBackBone(cfg)
#     B, S = 2, 10
#     x = torch.randn(B, S, cfg.n_embd)
#     out, memory, aux = model(inputs_embeds=x, output_hidden_states=True)
#     print(out.last_hidden_state.shape)   # [2, 10, 128]
#     print("mem layers:", memory.mem.shape, "cmem layers:", memory.compressed_mem.shape)
#     print("aux:", float(aux.detach()))

if __name__ == "__main__":
    cfg = CompressModelConfig(
        n_positions=64,
        n_embd=128,
        n_layer=4,
        n_head=8,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,

        mem_len=64,
        cmem_ratio=4,
        recon_attn_dropout=0.0,
        reconstruction_loss_weight=1.0,
    )
    model = CompressTransformerBackBone(cfg)

    # evaluation test
    model.eval()
    B, S = 2, cfg.n_positions
    memories = None
    for step in range(1, 4):
        x = torch.randn(B, S, cfg.n_embd)
        out, memories, aux = model(inputs_embeds=x, memories=memories)
        print(f"[Eval] step {step} -> mem {tuple(memories.mem.shape)}, cmem {tuple(memories.compressed_mem.shape)}, aux {float(aux)}")

    # train test
    model.train()
    memories = None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(1, 4):
        x = torch.randn(B, S, cfg.n_embd)
        out, memories, aux = model(inputs_embeds=x, memories=memories)
        main_loss = out.last_hidden_state.pow(2).mean()
        (main_loss + aux).backward()
        opt.step(); opt.zero_grad()
        print(f"[Train] step {step} -> aux {float(aux.detach()):.6f}")