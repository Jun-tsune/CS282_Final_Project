from typing import Optional
from src.models.transformer_backbone import *
from src.models.compressive_transformer_backbone import *
from src.utils.utils import ModelConfig, CompressModelConfig

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd, n_layer, n_head, resid_pdrop, embd_pdrop, attn_pdrop):
        super(TransformerModel, self).__init__()
        configuration = ModelConfig(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
        )
        self.name = f"transformer_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = TransformerBackBone(configuration)
        self._read_out = nn.Linear(n_embd, 1)
    
    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs
    
Memory = namedtuple("Memory", ["mem", "compressed_mem"])

class CompressiveTransformerModel(nn.Module):
    """
    A wrapper that mirrors TransformerModel but routes through a Compressive Transformer backbone.
    - Keeps the same x/y interleave (_combine)
    - Exposes optional recurrent memories and returns updated memories + aux loss
    """
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd,
        n_layer,
        n_head,
        resid_pdrop,
        embd_pdrop,
        attn_pdrop,
        # compressive-specific knobs (with sensible defaults)
        mem_len=None,            # default: n_positions in config if None
        cmem_ratio=4,
        cmem_len=None,           # default: mem_len // cmem_ratio if None
        recon_attn_dropout=0.0,
        reconstruction_loss_weight=1.0, # default: all layers if None (handled in backbone)
    ):
        super().__init__()
        # The sequence into backbone is length 2 * n_positions due to interleaving (xs, ys_wide)
        # so configure backbone with doubled window.
        configuration = CompressModelConfig(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            recon_attn_dropout=recon_attn_dropout,
            mem_len=mem_len if mem_len is not None else (2 * n_positions),
            cmem_len=cmem_len,
            cmem_ratio=cmem_ratio,
            reconstruction_loss_weight=reconstruction_loss_weight,
        )
        self.name = f"compress_transformer_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims

        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = CompressTransformerBackBone(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=xs_b.device, dtype=xs_b.dtype),
            ),
            dim=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)  # [B, P, 2, D]
        zs = zs.view(bsize, 2 * points, dim)        # [B, 2P, D]
        return zs

    def forward(
        self,
        xs,
        ys,
        inds=None,
        attention_mask=None,
        memories: Optional[Memory] = None,
        output_hidden_states: bool = False,
        enhanced_recurrence: bool = False,
    ):
        """
        xs: [B, S, D], ys: [B, S, 1]
        inds: which positions within S to return predictions for (like the baseline)
        attention_mask: optional [1,1,S,S] mask over the *xs-only* time steps; will be padded internally
        memories: optional (mem, compressed_mem) each shaped [Lmem, B, T, H]; pass from previous calls
        Returns:
          preds: [B, len(inds)] or [B, S]
          next_memories: Memory(mem=[L,B,T,H], compressed_mem=[L,B,Tc,H])
          aux_loss: scalar tensor (reconstruction MSE between old_mem attention and cmem attention)
          hidden_states (optional): same convention as baseline backbone when requested
        """
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.as_tensor(inds, device=ys.device)
            if int(inds.max()) >= ys.shape[1] or int(inds.min()) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)                 # [B, 2S, D]
        embeds = self._read_in(zs)                 # [B, 2S, H]

        # Backbone returns (SimpleNamespace, Memory, aux_loss)
        out, next_memories, aux_loss = self._backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            memories=memories,
            output_hidden_states=output_hidden_states,
            enhanced_recurrence=enhanced_recurrence,
        )

        # Read out on the full interleaved sequence then take only xs time steps (even indices)
        logits = self._read_out(out.last_hidden_state)   # [B, 2S, 1]
        preds = logits[:, ::2, 0][:, inds]               # [B, len(inds)]

        if output_hidden_states:
            return preds, next_memories, aux_loss, out.hidden_states
        return preds, next_memories, aux_loss
    


    
if __name__ == "__main__":
    B, S, H = 4, 56, 128
    x_ctx = torch.randn(B, S, H)
    y_ctx = torch.randn(B, S, 1)

    print("Transformer:")
    model = TransformerModel(
        n_dims=H,
        n_positions=512,
        n_embd=64,
        n_layer=2,
        n_head=4,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.1,
    )
    y_q_pred = model(x_ctx, y_ctx, inds=[0, 2])
    print("y_q_pred shape 1:", y_q_pred.shape)  # should be [B, len(inds)]
    print("y_q_pred:", y_q_pred)
    y_q_pred = model(x_ctx, y_ctx)
    print("y_q_pred shape 2:", y_q_pred.shape)  # should be [B, S]

    print("Compressive Transformer:")
    model = CompressiveTransformerModel(
        n_dims=H,
        n_positions=512,
        n_embd=64,
        n_layer=3,
        n_head=4,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        mem_len=256,
        cmem_ratio=4,
        recon_attn_dropout=0.0,
        reconstruction_loss_weight=1.0,
    )

    # 1) first pass with empty memories
    y_q_pred, mem1, aux1 = model(x_ctx, y_ctx, inds=[0, 2])
    print("y_q_pred shape 1:", y_q_pred.shape)  # [B, len(inds)]
    print("y_q_pred:", y_q_pred)
    print("mem layers:", mem1.mem.shape, "cmem layers:", mem1.compressed_mem.shape)
    print("aux:", float(aux1.detach()))

    # 2) second pass reusing memories (demonstrates recurrence)
    y_q_pred2, mem2, aux2 = model(x_ctx, y_ctx, memories=mem1)
    print("y_q_pred shape 2:", y_q_pred2.shape)  # [B, S]
    print("y_q_pred:", y_q_pred)
    print("next mem layers:", mem2.mem.shape, "next cmem layers:", mem2.compressed_mem.shape)