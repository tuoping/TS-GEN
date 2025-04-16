from .ema import ExponentialMovingAverage
from .logger import get_logger

logger = get_logger(__name__)

import torch, time
from torch import nn
import copy

from .model.equivariant_latent_model import EquivariantTransformer_dpm, Encoder_dpm, Processor, Decoder
from .wrapper import Wrapper

# Typing
from torch import Tensor
from typing import List, Optional, Tuple
from .transport.transport import create_transport, Sampler

class EquivariantMDGenWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        for key in [
            'cond_interval',
        ]:
            if not hasattr(args, key):
                setattr(args, key, False)
        
        num_species = args.num_species
        if args.design:
            num_scalar_out = 20
        else:
            num_scalar_out = 0
            
        self.model = EquivariantTransformer_dpm(
            encoder = Encoder_dpm(num_species, args.embed_dim, 4, args.edge_dim),
            processor = Processor(num_convs=args.num_convs, node_dim=args.embed_dim, num_heads=args.num_heads, ff_dim=args.ff_dim, edge_dim=args.edge_dim),
            decoder = Decoder(dim=args.embed_dim, num_scalar_out=num_scalar_out, num_vector_out=1),
            cutoff=args.cutoff,
            latent_dim=3,
            embed_dim=args.embed_dim,
            design=args.design,
        )

        self.transport = create_transport(
            args,
            args.path_type,
            args.prediction,
            None
        )
        self.transport_sampler = Sampler(self.transport)

        if not hasattr(args, 'ema'):
            args.ema = False
        if args.ema:
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=args.ema_decay
            )
            self.cached_weights = None
    '''
    def _prep_batch_latentgraph(self, batch):
        species = batch["species"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]

        latents = torch.cat([edge_index, edge_attr], -1)
        
        B, T, L, num_elem = species.shape
        h_loss_mask = batch["mask"].unsqueeze(-1).reshape(B,L,1)
        v_loss_mask = batch["v_mask"].reshape(B,L,3)
        loss_mask = torch.cat([h_loss_mask, v_loss_mask], -1)
        loss_mask = loss_mask.unsqueeze(1).expand(-1, T, -1, -1)

        B, T, D, _ = latents.shape
        assert _ == 6, f"latents shape should be (B, T, D, 4), but got {latents.shape}"
        ########
        cond_mask = torch.zeros(B, T, D, dtype=int, device=species.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        return {
            "species": species,
            "latents": latents,
            'loss_mask': loss_mask,
            'model_kwargs': {
                "aatype": species,
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                'x_cond_mask': cond_mask,
            }
        }
    '''

    def prep_batch(self, batch):
        species = batch["species"]
        latents = batch["x"]
    
        B, T, L, num_elem = species.shape
        v_loss_mask = batch["v_mask"].reshape(B,L,3)
        if self.args.design:
            h_loss_mask = batch["mask"].unsqueeze(-1).reshape(B,L,20)
            loss_mask = torch.cat([h_loss_mask, v_loss_mask], -1)
        else:
            loss_mask = v_loss_mask
        
        loss_mask = loss_mask.unsqueeze(1).expand(-1, T, -1, -1)

        B, T, L, _ = latents.shape
        assert _ == 3, f"latents shape should be (B, T, D, 3), but got {latents.shape}"
        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=species.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        return {
            "species": species,
            "latents": latents,
            'loss_mask': loss_mask,
            'model_kwargs': {
                "aatype": species,
                "cell": batch["cell"],
                "num_atoms": batch["num_atoms"],
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                'x_cond_mask': cond_mask,
            }
        }
    
    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        start1 = time.time()
        prep = self.prep_batch(batch)
        start = time.time()
        out_dict = self.transport.training_losses(
            model=self.model,
            x1=prep['latents'],
            aatype1=batch['species'],
            mask=prep['loss_mask'],
            model_kwargs=prep['model_kwargs']
        )
        self.log('model_dur', time.time() - start)
        loss = out_dict['loss']
        self.log('loss', loss)

        self.log('time', out_dict['t'])
        self.log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.log('name', ','.join(batch['name']))
        self.log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()
        return loss.mean()