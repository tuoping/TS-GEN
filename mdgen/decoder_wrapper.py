from .ema import ExponentialMovingAverage
from .logger import get_logger

logger = get_logger(__name__)

import torch, time
from torch import nn
import copy
import numpy as np
from functools import partial

from .model.equivariant_latent_model import TransformerDecoder
from .wrapper import Wrapper

# Typing
from torch import Tensor
from typing import List, Optional, Tuple
from .transport.transport import create_transport, Sampler

class DecoderWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        for key in [
            'cond_interval',
        ]:
            if not hasattr(args, key):
                setattr(args, key, False)
        
        num_scalar_out = self.args.num_species
        num_vector_out = 1
        
        self.model = TransformerDecoder(dim=args.embed_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out,      
                                 nhead=args.num_heads, 
                                 dim_feedforward=1024,
                                 activation='gelu',
                                 dropout=0.0,
                                 norm_first= True,
                                 bias= True,
                                 num_layers= 6,)
                 

        self.transport = create_transport(
            args,
            args.path_type,
            args.prediction,
            train_eps=1e-5,
            sample_eps=1e-5
        )
        self.transport_sampler = Sampler(self.transport)

        if not hasattr(args, 'ema'):
            args.ema = False
        if args.ema:
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=args.ema_decay
            )
            self.cached_weights = None

    def prep_batch(self, batch):
        if self.args.design:
            return self.prep_batch_species(batch)
        else:
            return self.prep_batch_x(batch)

    def prep_batch_species(self, batch):
        species = batch["species"]
        latents = batch["species"]
        x_now = batch["x"]
        
    
        B, T, L, num_elem = species.shape

        
        if self.args.design:
            loss_mask = batch["mask"]
            # loss_mask = torch.cat([h_loss_mask, v_loss_mask], -1)
            loss_mask = loss_mask
        else:
            v_loss_mask = batch["v_mask"]
            loss_mask = v_loss_mask


        B, T, L, _ = latents.shape
        assert _ == self.args.num_species, f"latents shape should be (B, T, D, self.args.num_species), but got {latents.shape}"
        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=species.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        return {
            "species": latents,
            "latents": latents,
            'loss_mask': loss_mask,
            'model_kwargs': {
                "cell": batch["cell"],
                "num_atoms": batch["num_atoms"],
                "conditions": None,
                "aatype": None,
                "x_latt": x_now,
            }
        }

    def prep_batch_x(self, batch):
        B, T, D, _ = batch["x"].shape
        mask = torch.ones_like(batch["x"], device=batch["x"].device)

        cond_mask = torch.zeros(B, T, dtype=int, device=batch["x"].device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        return {
            "v": batch["v"],
            "h": batch["h"],
            "species": batch["species"],
            "x": batch["x"],
            "frac_x": batch["frac_x"],
            "cell": batch["cell"],
            'loss_mask': mask,
        }
    
    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        start1 = time.time()
        prep = self.prep_batch(batch)
    
        start = time.time()
        B,T,N,_ = prep["h"].shape
        out_dict = self.model(prep["h"], prep["v"])
        L_aa = torch.nn.functional.cross_entropy(out_dict['aatype'].reshape(-1,self.args.num_species), prep['species'].reshape(-1,self.args.num_species).to(out_dict['aatype'].device).argmax(dim=-1), reduction='none').reshape(B,T,N).mean(dim=-1).mean(dim=-1)
        out_dict["pos"] -= out_dict["pos"].mean(dim=2, keepdim=True)
        batch["x"] -= batch["x"].mean(dim=2, keepdim=True)
        L_x = torch.nn.functional.mse_loss(out_dict['pos'], batch['x'], reduction='none').mean(dim=-1).mean(dim=-1).mean(dim=-1)
        L_frac = torch.nn.functional.mse_loss(out_dict['frac_pos'], batch['frac_x'], reduction='none').mean(dim=-1).mean(dim=-1).mean(dim=-1)
        L_cell_length =  torch.nn.functional.mse_loss(out_dict['cell'][...,:3], prep['cell'][...,:3].to(out_dict['aatype'].device), reduction='none').reshape(B,T,-1).mean(dim=-1).mean(dim=-1)/torch.sqrt(torch.tensor(N, dtype=torch.float))
        L_cell_angle =  torch.nn.functional.mse_loss(out_dict['cell'][...,3:], prep['cell'][...,3:].to(out_dict['aatype'].device), reduction='none').reshape(B,T,-1).mean(dim=-1).mean(dim=-1)/torch.sqrt(torch.tensor(N, dtype=torch.float))
        self.log('L_aa', L_aa)
        self.log('L_x', L_x)
        self.log('L_frac', L_frac)
        self.log('L_cell_length', L_cell_length)
        self.log('L_cell_angle', L_cell_angle)
        loss = L_aa + 0*L_x + 10*L_frac + L_cell_length + 10*L_cell_angle
        self.log('model_dur', time.time() - start)
        self.log('loss', loss)

        self.log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.log('name', ','.join(batch['name']))
        self.log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()
        
        return loss.mean()


    def inference(self, batch):

        prep = self.prep_batch(batch)

        latents = prep['latents']
        B, T, N, D = latents.shape

        if self.args.design:
            # zs_continuous = torch.randn(B, T, N, self.latent_dim - self.args.num_species, device=latents.device)
            zs_discrete = torch.distributions.Dirichlet(torch.ones(B, N, self.args.num_species, device=latents.device)).sample()
            zs_discrete = zs_discrete[:, None].expand(-1, T, -1, -1)
            # zs = torch.cat([zs_continuous, zs_discrete], -1)
            zs = zs_discrete

            x1 = prep['latents']
            x_d = torch.zeros(x1.shape[0], x1.shape[1], x1.shape[2], self.args.num_species, device=self.device)
            xt = torch.cat([x1, x_d], dim=-1)
            logits = self.model.forward_inference(xt, torch.ones(B, device=self.device),
                                                  **prep['model_kwargs'])
            aa_out = torch.argmax(logits, -1)
            # aa_out = logits
            vector_out = prep["model_kwargs"]["x_latt"]
            return vector_out, aa_out
        else:
            zs = torch.randn(B, T, N, D, device=self.device)

        sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method)  # default to ode
        # sample_fn = self.transport_sampler.sample_sde(sampling_method=self.args.sampling_method, num_steps=self.args.inference_steps)  # default to ode

        samples = sample_fn(
            zs,
            partial(self.model.forward_inference, **prep['model_kwargs'])
        )[-1]
        
        if self.args.design:
            # vector_out = samples[..., :-self.args.num_species]
            vector_out = prep["model_kwargs"]["x_now"]
            logits = samples[..., -self.args.num_species:]
        else:
            vector_out = samples

        if self.args.design:
            aa_out = torch.argmax(logits, -1)
            # aa_out = logits
        else:
            aa_out = torch.argmax(batch['species'], -1)
            # aa_out = batch['species']
        return vector_out, aa_out