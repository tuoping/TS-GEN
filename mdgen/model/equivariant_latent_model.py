# --------------------------------------------------------
# Adapted from: https://github.com/LLNL/graphite
# --------------------------------------------------------

import torch
from torch import nn
from torch_geometric.utils import scatter
import copy

from .nn.mlp import MLP
from .nn.convs import EquivariantTransformerLayer
from .nn.basis import GaussianRandomFourierFeatures

# Typing
from torch import Tensor
from typing import List, Optional, Tuple


class Encoder(nn.Module):
    def __init__(self, num_species: int, node_dim: int, init_edge_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.num_species   = num_species
        self.node_dim      = node_dim
        self.init_edge_dim = init_edge_dim
        self.edge_dim      = edge_dim
        
        # self.embed_atom = nn.Embedding(num_species, node_dim)
        self.embed_atom = nn.Linear(5, node_dim)
        self.embed_bond = MLP([init_edge_dim, edge_dim, edge_dim], act=nn.SiLU())
        self.phi_s = MLP([node_dim*2 + edge_dim, edge_dim, node_dim], act=nn.SiLU())
        self.phi_h = MLP([node_dim*2,            edge_dim, node_dim], act=nn.SiLU())
        self.phi_v = MLP([node_dim*2 + edge_dim, edge_dim, node_dim], act=nn.SiLU())

    def forward(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        num_nodes = len(species)
        i = edge_index[0]
        j = edge_index[1]
        
        # Embed node and edge features
        f         = self.embed_atom(species)
        edge_attr = self.embed_bond(edge_attr)

        # Convolve node features
        e  = torch.cat([f[i], f[j], edge_attr], dim=-1)
        h0 = self.phi_h(torch.cat([
            f, scatter(self.phi_s(e) * f[i], index=j, dim=0, dim_size=num_nodes)
        ], dim=-1))

        # Initialize vector features
        v0 = scatter(edge_vec[:, None, :] * self.phi_v(e)[:, :, None], index=j, dim=0, dim_size=num_nodes)
        return h0, v0, edge_attr


class Encoder_dpm(Encoder):
    def __init__(self, num_species: int, node_dim: int, init_edge_dim: int, edge_dim: int, input_dim: int) -> None:
        super().__init__(num_species, node_dim, init_edge_dim, edge_dim)
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(node_dim, input_dim=input_dim),
            MLP([node_dim, edge_dim, node_dim], act=nn.SiLU()),
        )

    def forward(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_nodes = len(species)
        i = edge_index[0]
        j = edge_index[1]

        # Embed node and edge features
        f         = self.embed_atom(species)
        edge_attr = self.embed_bond(edge_attr)

        # Convolve node features
        e  = torch.cat([f[i], f[j], edge_attr], dim=-1)
        h0 = self.phi_h(torch.cat([
            f, scatter(self.phi_s(e) * f[i], index=j, dim=0, dim_size=num_nodes)
        ], dim=-1))

        # Initialize vector features
        v0 = scatter(edge_vec[:, None, :] * self.phi_v(e)[:, :, None], index=j, dim=0, dim_size=num_nodes)

        # Add time embedding to node features
        h0 = h0 + self.embed_time(t)
        return h0, v0, edge_attr


class Processor(nn.Module):
    def __init__(self, num_convs: int, node_dim: int, num_heads: int, ff_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.num_convs = num_convs
        self.node_dim  = node_dim
        self.ff_dim    = ff_dim
        self.edge_dim  = edge_dim
        self.convs = nn.ModuleList(
            [copy.deepcopy(EquivariantTransformerLayer(node_dim, num_heads, ff_dim, edge_dim)) for _ in range(num_convs)]
        )

    def forward(self, h: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_len: Tensor) -> Tuple[Tensor, Tensor]:
        for conv in self.convs:
            h, v = conv(h, v, edge_index, edge_attr, edge_len)
        return h, v


class Decoder(nn.Module):
    def __init__(self, dim: int, num_scalar_out: int, num_vector_out: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_scalar_out = num_scalar_out
        self.num_vector_out = num_vector_out
        self.Oh = nn.Parameter(torch.randn(dim, num_scalar_out))
        self.Ov = nn.Parameter(torch.randn(dim, num_vector_out))

    def forward(self, h:Tensor, v: Tensor) -> Tensor:
        h_ = h @ self.Oh
        v_out = torch.einsum('ndi, df -> nfi', v, self.Ov)
        # return h_out, v_out.squeeze()
        if h_.shape[-1] >= 5:
            h_out_1 = h_[...,:-5]
            h_out_2 = torch.nn.functional.softmax(h_[...,-5:], dim=-1)
            return torch.hstack([h_out_1, h_out_2]), v_out.squeeze()
        else:
            return h_, v_out.squeeze()
        

    def extra_repr(self) -> str:
        return f'(Oh): tensor({list(self.Oh.shape)}, requires_grad={self.Oh.requires_grad}) \n' \
             + f'(Ov): tensor({list(self.Ov.shape)}, requires_grad={self.Ov.requires_grad})'



class EquivariantTransformer(nn.Module):
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        h, v, edge_attr = self.encoder(species, edge_index, edge_attr, edge_vec)
        h, v = self.processor(h, v, edge_index, edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
        return self.decoder(h, v)


from .utils.data_utils import (
    frac_to_cart_coords_with_lattice,
    get_pbc_distances,
    lattice_params_to_matrix_torch,
    radius_graph_pbc,
)


import numpy as np
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class EquivariantTransformer_dpm(EquivariantTransformer):
    def __init__(self, encoder, processor, decoder, cutoff, latent_dim, embed_dim, otf_graph = True, design=False, potential_model=False, abs_time_emb=False, num_frames=None):
        super().__init__(encoder, processor, decoder)
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.design = design
        self.potential_model = potential_model
        self.max_num_neighbors_threshold = 50
        self.max_cell_images_per_dim = 5

        cond_dim = latent_dim
        self.cond_dim = cond_dim
        self.cond_to_emb = nn.Linear(cond_dim, embed_dim)
        self.mask_to_emb = nn.Embedding(cond_dim, embed_dim)
        self.abs_time_emb = abs_time_emb
        if abs_time_emb:
            self.register_buffer('time_embed',
                                 nn.Parameter(torch.zeros(1, num_frames, embed_dim), requires_grad=False))
            time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], np.arange(num_frames))
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

    def _graph_forward(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor, t: Tensor, out_cond=None,) -> Tuple[Tensor, Tensor]:
        h, v, edge_attr = self.encoder(species, edge_index, edge_attr, edge_vec, t)
        if self.abs_time_emb:
            h = h + self.time_embed[:, :, None]
        if out_cond is not None:
            # edge_index_cond = out_cond["edge_index"]
            # edge_len_cond = out_cond["distances"]
            # edge_vec_cond = out_cond["distance_vec"]
            # edge_attr_cond = torch.hstack([edge_vec_cond, edge_len_cond.view(-1, 1)])
            # species_cond = out_cond["species"]
            # h_cond, v_cond, edge_attr_cond = self.encoder(
            #     species_cond.view(-1,5), 
            #     edge_index_cond, edge_attr_cond, edge_vec_cond, 
            #     torch.zeros([*species_cond.shape[:-1],1], device=species_cond.device).reshape(-1,1)
            #     )
            # h = h + self.cond_to_emb(h_cond)*(out_cond["mask"].reshape(-1,1)) # + self.mask_to_emb(out_cond["mask"])
            h = h + self.cond_to_emb(out_cond["x"]) + self.mask_to_emb(out_cond["mask"])
        h, v = self.processor(h, v, edge_index, edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
        return self.decoder(h, v)

    
    def get_processed_var(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None, 
                aatype=None):
        assert cell is not None
        B, T, N, _ = x.shape
        assert t.shape == (B,)
        if self.otf_graph:
            self.edge_index, self.to_jimages, self.num_bonds = radius_graph_pbc(
                cart_coords=x.view(-1, 3),
                lattice=cell.view(-1, 3, 3),
                num_atoms=num_atoms.view(-1),
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_num_neighbors_threshold,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )
            # if conditions is not None:
            #     self.edge_index_cond, self.to_jimages_cond, self.num_bonds_cond = radius_graph_pbc(
            #         cart_coords=conditions["x"].view(-1, 3),
            #         lattice=conditions["cell"].view(-1, 3, 3),
            #         num_atoms=conditions["num_atoms"].view(-1),
            #         radius=self.cutoff,
            #         max_num_neighbors_threshold=self.max_num_neighbors_threshold,
            #         max_cell_images_per_dim=self.max_cell_images_per_dim,
            #     )
            # self.otf_graph = False

        out = get_pbc_distances(
            x.view(-1, 3),
            self.edge_index,
            cell.view(-1, 3, 3),
            self.to_jimages,
            num_atoms.view(-1),
            self.num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        # if conditions is not None:
        #     out_cond = get_pbc_distances(
        #         conditions["x"].view(-1, 3),
        #         self.edge_index_cond,
        #         conditions["cell"].view(-1, 3, 3),
        #         self.to_jimages_cond,
        #         conditions["num_atoms"].view(-1),
        #         self.num_bonds_cond,
        #         coord_is_cart=True,
        #         return_offsets=True,
        #         return_distance_vec=True,
        #     )
        #     out_cond["species"] = conditions["species"]
        #     out_cond["mask"] = conditions["mask"]
        edge_index = out["edge_index"]
        edge_len = out["distances"]
        edge_vec = out["distance_vec"]
        edge_attr = torch.hstack([edge_vec, edge_len.view(-1, 1)])

        t = t.unsqueeze(-1).unsqueeze(1).expand(-1,T,-1).unsqueeze(2).expand(-1,-1,N,-1)
        if aatype is not None:
            species = aatype
        else:
            aatype = torch.zeros([B,T,N], dtype=torch.long, device=x.device)
            species = torch.nn.functional.one_hot(aatype, num_classes=5, dtype=torch.float)
            
        h, v, edge_attr = self.encoder(species.reshape(-1,5), edge_index, edge_attr, edge_vec, t.reshape(-1,1))
        h, v = self.processor(h, v, edge_index, edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
        return h, v
    
    def get_encoded_var(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None, 
                aatype=None):
        assert cell is not None
        B, T, N, _ = x.shape
        assert t.shape == (B,)
        if self.otf_graph:
            self.edge_index, self.to_jimages, self.num_bonds = radius_graph_pbc(
                cart_coords=x.view(-1, 3),
                lattice=cell.view(-1, 3, 3),
                num_atoms=num_atoms.view(-1),
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_num_neighbors_threshold,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )
            # if conditions is not None:
            #     self.edge_index_cond, self.to_jimages_cond, self.num_bonds_cond = radius_graph_pbc(
            #         cart_coords=conditions["x"].view(-1, 3),
            #         lattice=conditions["cell"].view(-1, 3, 3),
            #         num_atoms=conditions["num_atoms"].view(-1),
            #         radius=self.cutoff,
            #         max_num_neighbors_threshold=self.max_num_neighbors_threshold,
            #         max_cell_images_per_dim=self.max_cell_images_per_dim,
            #     )
            # self.otf_graph = False

        out = get_pbc_distances(
            x.view(-1, 3),
            self.edge_index,
            cell.view(-1, 3, 3),
            self.to_jimages,
            num_atoms.view(-1),
            self.num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        # if conditions is not None:
        #     out_cond = get_pbc_distances(
        #         conditions["x"].view(-1, 3),
        #         self.edge_index_cond,
        #         conditions["cell"].view(-1, 3, 3),
        #         self.to_jimages_cond,
        #         conditions["num_atoms"].view(-1),
        #         self.num_bonds_cond,
        #         coord_is_cart=True,
        #         return_offsets=True,
        #         return_distance_vec=True,
        #     )
        #     out_cond["species"] = conditions["species"]
        #     out_cond["mask"] = conditions["mask"]
        edge_index = out["edge_index"]
        edge_len = out["distances"]
        edge_vec = out["distance_vec"]
        edge_attr = torch.hstack([edge_vec, edge_len.view(-1, 1)])

        t = t.unsqueeze(-1).unsqueeze(1).expand(-1,T,-1).unsqueeze(2).expand(-1,-1,N,-1)
        if aatype is not None:
            species = aatype
        else:
            aatype = torch.zeros([B,T,N], dtype=torch.long, device=x.device)
            species = torch.nn.functional.one_hot(aatype, num_classes=5, dtype=torch.float)
            
        h, v, edge_attr = self.encoder(species.reshape(-1,5), edge_index, edge_attr, edge_vec, t.reshape(-1,1))
        
        return h, v, edge_attr

    def forward_encoder(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x_latt=None):
        if self.design:
            x_ = x_latt
            aatype_ = x
            scaler_out = self.get_encoded_var(x_, t, cell, num_atoms, conditions, aatype_)
            return scaler_out
        else:
            vector_out = self.get_encoded_var(x, t, cell, num_atoms, conditions, aatype)
            return vector_out
        

    def forward_processor(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x_latt=None):
        if self.design:
            x_ = x_latt
            aatype_ = x
            scaler_out = self.get_processed_var(x_, t, cell, num_atoms, conditions, aatype_)
            return scaler_out
        else:
            vector_out = self.get_processed_var(x, t, cell, num_atoms, conditions, aatype)
            return vector_out
    
    def inference(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None, 
                aatype=None):
        assert cell is not None
        B, T, N, _ = x.shape
        assert t.shape == (B,)

        if self.otf_graph:
            self.edge_index, self.to_jimages, self.num_bonds = radius_graph_pbc(
                cart_coords=x.view(-1, 3),
                lattice=cell.view(-1, 3, 3),
                num_atoms=num_atoms.view(-1),
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_num_neighbors_threshold,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )
            # if conditions is not None:
            #     self.edge_index_cond, self.to_jimages_cond, self.num_bonds_cond = radius_graph_pbc(
            #         cart_coords=conditions["x"].view(-1, 3),
            #         lattice=conditions["cell"].view(-1, 3, 3),
            #         num_atoms=conditions["num_atoms"].view(-1),
            #         radius=self.cutoff,
            #         max_num_neighbors_threshold=self.max_num_neighbors_threshold,
            #         max_cell_images_per_dim=self.max_cell_images_per_dim,
            #     )
            # self.otf_graph = False

        out = get_pbc_distances(
            x.view(-1, 3),
            self.edge_index,
            cell.view(-1, 3, 3),
            self.to_jimages,
            num_atoms.view(-1),
            self.num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        # if conditions is not None:
        #     out_cond = get_pbc_distances(
        #         conditions["x"].view(-1, 3),
        #         self.edge_index_cond,
        #         conditions["cell"].view(-1, 3, 3),
        #         self.to_jimages_cond,
        #         conditions["num_atoms"].view(-1),
        #         self.num_bonds_cond,
        #         coord_is_cart=True,
        #         return_offsets=True,
        #         return_distance_vec=True,
        #     )
        #     out_cond["species"] = conditions["species"]
        #     out_cond["mask"] = conditions["mask"]
        edge_index = out["edge_index"]
        edge_len = out["distances"]
        edge_vec = out["distance_vec"]
        edge_attr = torch.hstack([edge_vec, edge_len.view(-1, 1)])

        t = t.unsqueeze(-1).unsqueeze(1).expand(-1,T,-1).unsqueeze(2).expand(-1,-1,N,-1)
        if aatype is not None:
            species = aatype
        else:
            aatype = torch.zeros([B,T,N], dtype=torch.long, device=x.device)
            species = torch.nn.functional.one_hot(aatype, num_classes=5, dtype=torch.float)
            
        
        scaler_out, vector_out = self._graph_forward(species.reshape(-1,5), edge_index, edge_attr, edge_vec, t.reshape(-1,1), conditions)
        if self.design:
            # return torch.hstack([vector_out, scaler_out]).view(B, T, N, -1)
            return scaler_out.view(B, T, N, -1)
        elif self.potential_model:
            return vector_out.reshape(B, T, N, -1)
        else:
            return vector_out.reshape(B, T, N, -1)
        
    def forward(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x_latt=None, x1=None, v_mask=None):
        if self.design:
            x_ = x_latt
            aatype_ = x
            if not self.potential_model:
                x_ = x_*v_mask+x1*(1-v_mask)
            scaler_out = self.inference(x_, t, cell, num_atoms, conditions, aatype_)
            return scaler_out*v_mask
        else:
            if not self.potential_model:
                x = x*v_mask+x1*(1-v_mask)
            vector_out = self.inference(x, t, cell, num_atoms, conditions, aatype)
            return vector_out*v_mask

    def forward_inference(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x_latt=None, x1=None, v_mask=None):
        if self.design:
            x_ = x_latt
            aatype_ = x
            if not self.potential_model:
                x_ = x_*v_mask+x1*(1-v_mask)
            scaler_out = self.inference(x_, t, cell, num_atoms, conditions, aatype_)
            return scaler_out*v_mask
        else:
            if not self.potential_model:
                x = x*v_mask+x1*(1-v_mask)
            vector_out = self.inference(x, t, cell, num_atoms, conditions, aatype)
            return vector_out*v_mask
    

class TransformerDecoder(nn.Module):
    def __init__(self, dim: int, num_scalar_out: int, num_vector_out: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_scalar_out = num_scalar_out
        self.num_vector_out = num_vector_out
        self.Oh = nn.Parameter(torch.randn(dim, num_scalar_out))
        self.Ov = nn.Parameter(torch.randn(dim, num_vector_out))

    def forward(self, h:Tensor, v: Tensor) -> Tensor:
        h_ = h @ self.Oh
        v_out = torch.einsum('ndi, df -> nfi', v, self.Ov)
        # return h_out, v_out.squeeze()
        assert h_.shape[-1] == 5
        h_out = torch.nn.functional.softmax(h_[...,-5:], dim=-1)
        return h_out, v_out.squeeze()

        

    def extra_repr(self) -> str:
        return f'(Oh): tensor({list(self.Oh.shape)}, requires_grad={self.Oh.requires_grad}) \n' \
             + f'(Ov): tensor({list(self.Ov.shape)}, requires_grad={self.Ov.requires_grad})'


"""
Adapted from: https://github.com/facebookresearch/all-atom-diffusion-transformer/
"""
'''
import math
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum length

    Returns:
        positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class TransformerDecoder(nn.Module):
    """Transformer decoder as part of pure Transformer-based VAEs.

    See src/models/encoders/transformer.py for documentation.
    """

    def __init__(
        self,
        max_num_elements=100,
        d_model: int = 1024,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 6,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers

        activation = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

        self.atom_types_head = nn.Linear(d_model, max_num_elements, bias=True)
        self.pos_head = nn.Linear(d_model, 3, bias=False)
        self.frac_coords_head = nn.Linear(d_model, 3, bias=False)
        self.lattice_head = nn.Linear(d_model, 6, bias=False)

    def forward(self, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoded_batch: Dict with the following attributes:
                x (torch.Tensor): Encoded batch of atomic environments
                num_atoms (torch.Tensor): Number of atoms in each molecular environment
                batch (torch.Tensor): Batch index for each atom
                token_idx (torch.Tensor): Token index for each atom
        """
        x = encoded_batch["x"]

        # Positional embedding
        x += get_index_embedding(encoded_batch["token_idx"], self.d_model)

        # Convert from PyG batch to dense batch with padding
        x, token_mask = to_dense_batch(x, encoded_batch["batch"])

        # Transformer forward pass
        x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]

        # Global pooling: (n, d) -> (bsz, d)
        x_global = scatter(x, encoded_batch["batch"], dim=0, reduce="mean")

        # Atomic type prediction head
        atom_types_out = self.atom_types_head(x)

        # Lattice lengths and angles prediction head
        lattices_out = self.lattice_head(x_global)

        # Fractional coordinates prediction head
        frac_coords_out = self.frac_coords_head(x)

        # Cartesian coordinates prediction head
        pos_out = self.pos_head(x)

        return {
            "atom_types": atom_types_out,
            "lattices": lattices_out,
            "lengths": lattices_out[:, :3],
            "angles": lattices_out[:, 3:],
            "frac_coords": frac_coords_out,
            "pos": pos_out,
        }
'''