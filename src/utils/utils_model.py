class ModelConfig:
    """A config object for transformer model configuration usage."""
    def __init__(
        self,
        n_positions=512,
        n_embd=256,
        n_layer=2,
        n_head=8,
        resid_pdrop=0.0,
        embd_pdrop=0.1,
        attn_pdrop=0.0,
    ):
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop # dropout for residual connections
        self.attn_pdrop = attn_pdrop # dropout for attention weights
        self.embd_pdrop = embd_pdrop # dropout for input embeddings