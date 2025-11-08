from src.models.transformer_backbone import *

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
    




    
if __name__ == "__main__":
    B, S, H = 8, 56, 128
    x_ctx = torch.randn(B, S, H)
    y_ctx = torch.randn(B, S, 1)

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
    y_q_pred = model(x_ctx, y_ctx)
    print("y_q_pred shape 2:", y_q_pred.shape)  # should be [B, S]