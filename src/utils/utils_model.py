from src.models.full_models import TransformerModel
from src.utils.utils import ModelConfig

def build_model(conf: ModelConfig):
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
    else:
        raise NotImplementedError

    return model