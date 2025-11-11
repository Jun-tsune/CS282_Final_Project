from src.models.full_models import CompressiveTransformerModel, TransformerModel
from src.utils.utils import CompressModelConfig

def build_model(conf: CompressModelConfig):
    if conf.model_family == "transformer":
        model = TransformerModel(
            n_dims=conf.n_dims, # input feature dimension
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            resid_pdrop=conf.resid_pdrop,
            embd_pdrop=conf.embd_pdrop,
            attn_pdrop=conf.attn_pdrop,
        )
    elif conf.model_family == "compressive":
        model = CompressiveTransformerModel(
            n_dims=conf.n_dims, # input feature dimension
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            resid_pdrop=conf.resid_pdrop,
            embd_pdrop=conf.embd_pdrop,
            attn_pdrop=conf.attn_pdrop,
            # compressive-specific knobs (with sensible defaults)
            mem_len=conf.mem_len,            # default: n_positions in config if None
            cmem_ratio=conf.cmem_ratio,
            cmem_len=conf.cmem_len,           # default: mem_len // cmem_ratio if None
            recon_attn_dropout=conf.recon_attn_dropout,
            reconstruction_loss_weight=conf.reconstruction_loss_weight,
        )
    else:
        raise NotImplementedError

    return model