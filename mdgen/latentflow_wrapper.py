from .ema import ExponentialMovingAverage
from .logger import get_logger

logger = get_logger(__name__)

import torch, time
from torch import nn
import copy
import numpy as np
from functools import partial

from .model.equivariant_latent_model import TransformerProcessor
from .wrapper import Wrapper

# Typing
from torch import Tensor
from typing import List, Optional, Tuple
from .transport.transport import create_transport, Sampler

class LatentGenWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        for key in [
            'cond_interval',
        ]:
            if not hasattr(args, key):
                setattr(args, key, False)

        num_scalar_out = self.args.num_species
        num_vector_out= 1 

        self.model = TransformerProcessor(dim=args.embed_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out,      
                                 nhead=args.num_heads, 
                                 dim_feedforward=1024,
                                 activation='gelu',
                                 dropout=0.0,
                                 norm_first= True,
                                 bias= True,
                                 num_layers= 6,
                                 node_dim=args.embed_dim,
                                 edge_dim=args.edge_dim,
                                 input_dim=1,)

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
        h = batch["h"].unsqueeze(-1)
        v = batch["v"]
        x = torch.concatenate([h,v], -1)
        B, T, L, D, _ = x.shape
        mask = torch.ones_like(x, device=x.device)
        mask[:,:,:,:,1:] = 0 # mask out the last 3 channels (v)
        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=x.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        return {
            "latents": x,
            "loss_mask": mask,
            "model_kwargs": {}
        }

    def prep_batch_x(self, batch):
        h = batch["h"].unsqueeze(-1)
        v = batch["v"]
        x = torch.concatenate([h,v], -1)
        B, T, L, D, _ = x.shape
        mask = torch.ones_like(x, device=x.device)
        mask[:,:,:,:,0] = 0 # mask out the first channel (h)
        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=x.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        return {
            "latents": x,
            "loss_mask": mask,
            "model_kwargs": {
                "x1": x,
                "v_mask": mask > 0,
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
            aatype1=None,
            mask=prep['loss_mask'],
            model_kwargs=prep["model_kwargs"],
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


    def inference(self, batch):

        prep = self.prep_batch(batch)

        latents = prep['latents']
        B, T, N, D, K = latents.shape

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
            zs = torch.randn(B, T, N, D, K, device=self.device)

        sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method)  # default to ode
        # sample_fn = self.transport_sampler.sample_sde(sampling_method=self.args.sampling_method, num_steps=self.args.inference_steps)  # default to ode

        samples = sample_fn(
            zs,
            partial(self.model.forward_inference, **prep['model_kwargs']),
        )[-1]
        
        if self.args.design:
            # vector_out = samples[..., :-self.args.num_species]
            vector_out = prep["model_kwargs"]["x_now"]
            logits = samples[..., -self.args.num_species:]
        else:
            vector_out = samples*prep['loss_mask'] + prep['latents']*(1-prep["loss_mask"])

        if self.args.design:
            aa_out = torch.argmax(logits, -1)
            # aa_out = logits
        else:
            aa_out = torch.argmax(batch['species'], -1)
            # aa_out = batch['species']
        return vector_out, aa_out