import torch.nn as nn

from lzero.model.exts.activations.swiglu import SwiGLU


def build_feed_forward_by_type(type: str, in_features: int, hidden_features: int, out_features: int) -> nn.Module:
    if type == 'base':
        return nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_features, out_features),
        )
    elif type == 'swiglu':
        assert (hidden_features * 2) % 3 == 0 # keep number of parameters are same

        hidden_features = (hidden_features * 2) // 3

        return nn.Sequential(
            nn.Linear(in_features, hidden_features * 2),
            SwiGLU(),
            nn.Linear(hidden_features, out_features),
        )
    else:
        raise Exception('not supported feed forward type: ' + type)
