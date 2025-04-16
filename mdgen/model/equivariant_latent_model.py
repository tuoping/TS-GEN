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
        self.embed_atom = nn.Linear(20, node_dim)
        self.embed_bond = MLP([init_edge_dim, edge_dim, edge_dim], act=nn.SiLU())
        self.phi_s = MLP([node_dim*2 + edge_dim, node_dim, node_dim], act=nn.SiLU())
        self.phi_h = MLP([node_dim*2,            node_dim, node_dim], act=nn.SiLU())
        self.phi_v = MLP([node_dim*2 + edge_dim, node_dim, node_dim], act=nn.SiLU())

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
    def __init__(self, num_species: int, node_dim: int, init_edge_dim: int, edge_dim: int) -> None:
        super().__init__(num_species, node_dim, init_edge_dim, edge_dim)
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(node_dim, input_dim=1),
            MLP([node_dim, node_dim, node_dim], act=nn.SiLU()),
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
        h_out_1 = h_[:,:-20]
        h_out_2 = torch.nn.functional.softmax(h_[:,-20:], dim=1)
        return torch.hstack([h_out_1, h_out_2]), v_out.squeeze()

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
class EquivariantTransformer_dpm(EquivariantTransformer):
    def __init__(self, encoder, processor, decoder, cutoff, latent_dim, embed_dim, otf_graph = True, design=False):
        super().__init__(encoder, processor, decoder)
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.design = design
        self.max_num_neighbors_threshold = 50
        self.max_cell_images_per_dim = 5

        cond_dim = latent_dim
        if self.design: cond_dim -= 20
        self.cond_to_emb = nn.Linear(cond_dim, embed_dim)
        self.mask_to_emb = nn.Embedding(2, embed_dim)

    def _graph_forward(self, species: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vec: Tensor, t: Tensor, x_cond=None, x_cond_mask=None,) -> Tuple[Tensor, Tensor]:
        h, v, edge_attr = self.encoder(species, edge_index, edge_attr, edge_vec, t)
        if x_cond is not None:
            h = h + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)  # token has cond g, tau
        h, v = self.processor(h, v, edge_index, edge_attr, edge_len=torch.linalg.norm(edge_vec, dim=1, keepdim=True))
        return self.decoder(h, v)
    
    def forward(self, x: Tensor, t: Tensor, 
                cell=None, 
                num_atoms=None,
                x_cond=None, x_cond_mask=None, 
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
            self.otf_graph = False

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

        edge_index = out["edge_index"]
        edge_len = out["distances"]
        edge_vec = out["distance_vec"]
        edge_attr = torch.hstack([edge_vec, edge_len.view(-1, 1)])

        if aatype is not None:
            species = aatype.view(-1,20)
        else:
            aatype = torch.zeros([N], dtype=torch.long, device=x.device)
            species = torch.nn.functional.one_hot(aatype, num_classes=20, dtype=torch.float)
            
        
        scaler_out, vector_out = self._graph_forward(species, edge_index, edge_attr, edge_vec, t, x_cond=x_cond.view(-1,3), x_cond_mask=x_cond_mask.view(-1))
        if self.design:
            return torch.hstack([vector_out, scaler_out]).view(B, T, N, -1)
        else:
            return vector_out.view(B, T, N, -1)