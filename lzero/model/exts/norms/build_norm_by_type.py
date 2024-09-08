import torch.nn as nn

from lzero.model.exts.norms.rms_norm import RMSNorm


def build_norm_by_type(type: str, in_features: int) -> nn.Module:
    if type == 'LN':
        return nn.LayerNorm(in_features)
    elif type == 'RMS':
        return RMSNorm(in_features)
    else:
        raise Exception('not supported norm type: ' + type)
