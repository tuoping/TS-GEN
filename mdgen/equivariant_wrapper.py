import torch
from torch import nn
import copy

from .model.nn.models.equivariant_transformer import EquivariantTransformer, EquivariantTransformer_dpm
from .wrapper import Wrapper

# Typing
from torch import Tensor
from typing import List, Optional, Tuple

class EquivariantMDGenWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        for key in [
            'cond_interval',
        ]:
            if not hasattr(args, key):
                setattr(args, key, False)