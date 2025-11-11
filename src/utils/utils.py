
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

class CompressModelConfig:
    """A config object for compressive transformer model configuration usage."""
    def __init__(
        self,
        n_positions=512,
        n_embd=256,
        n_layer=2,
        n_head=8,
        resid_pdrop=0.0,
        embd_pdrop=0.1,
        attn_pdrop=0.0,
        # Compressive paras
        recon_attn_dropout=0.0,
        mem_len=512, # default equals n_positions
        cmem_len=None,  # default mem_len // cmem_ratio
        cmem_ratio=4,
        reconstruction_loss_weight=1.0
    ):
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop # dropout for residual connections
        self.attn_pdrop = attn_pdrop # dropout for attention weights
        self.embd_pdrop = embd_pdrop # dropout for input embeddings
        self.recon_attn_dropout = recon_attn_dropout
        self.mem_len = mem_len
        self.cmem_len = mem_len // cmem_ratio
        self.cmem_ratio = cmem_ratio
        self.reconstruction_loss_weight = reconstruction_loss_weight