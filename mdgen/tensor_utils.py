# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import List

import torch
import torch.nn as nn


def add(m1, m2, inplace):
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if(not inplace):
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def pts_to_distogram(pts, min_bin=2.3125, max_bin=21.6875, no_bins=64):
    boundaries = torch.linspace(
        min_bin, max_bin, no_bins - 1, device=pts.device
    )
    dists = torch.sqrt(
        torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1)
    )
    return torch.bucketize(dists, boundaries)


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x, v_bins):
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    # ranges = []
    # for i, s in enumerate(data.shape[:no_batch_dims]):
    #     r = torch.arange(s)
    #     r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
    #     ranges.append(r)
    # remaining_dims = [
    #     slice(None) for _ in range(len(data.shape) - no_batch_dims)
    # ]
    # remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    # ranges.extend(remaining_dims)
    # return data[ranges]
    shape_d = data.shape
    shape_i = inds.shape
    _data = data
    if len(shape_d)+dim < no_batch_dims:
        _data = _data.view(-1, *(1,)*(no_batch_dims-len(shape_d)-dim), *shape_d[1:])
    shape_d = _data.shape
    for n in range(no_batch_dims):
        if _data.shape[n] < inds.shape[n]:

            _data = _data.expand(*(-1,)*(n), shape_i[n], *(-1,)*(len(shape_d)-n-1))
    if dim == -2:
        _inds = inds.unsqueeze(-1).expand(*(-1,)*(len(shape_i)), _data.shape[-1])
        _inds = _inds.expand(_data.shape[0], *(-1,)*(len(shape_i)))
        # return _data[...,inds,:]
        return torch.gather(_data, dim=len(shape_i)-1, index=_inds).squeeze(len(shape_i)-1)
    else:
        _inds = inds.expand(_data.shape[0], *(-1,)*(len(shape_i)-1))
        # return _data[...,inds]
        return torch.gather(_data, dim=len(shape_i)-1, index=_inds).squeeze(len(shape_i)-1)


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Tree of type {type(tree)} not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)