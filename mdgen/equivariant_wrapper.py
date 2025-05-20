from .ema import ExponentialMovingAverage
from .logger import get_logger

logger = get_logger(__name__)

import torch, time
from torch import nn
import copy
import numpy as np
from functools import partial

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
            num_scalar_out = 5
            num_vector_out=0
            latent_dim = 5
        elif args.potential_model:
            num_scalar_out = 1
            latent_dim = 3
            num_vector_out = 0
        else:
            num_scalar_out = 0
            num_vector_out=1
            latent_dim = 3
        
        self.model = EquivariantTransformer_dpm(
            encoder = Encoder_dpm(num_species, args.embed_dim, 4, args.edge_dim, input_dim=1),
            processor = Processor(num_convs=args.num_convs, node_dim=args.embed_dim, num_heads=args.num_heads, ff_dim=args.ff_dim, edge_dim=args.edge_dim),
            decoder = Decoder(dim=args.embed_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out),
            cutoff=args.cutoff,
            latent_dim=latent_dim,
            embed_dim=args.embed_dim,
            design=args.design,
            potential_model = args.potential_model
        )

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
        assert _ == 5, f"latents shape should be (B, T, D, 5), but got {latents.shape}"
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
        species = batch["species"]
        latents = batch["x"]
        # rdf = batch["RDF"]
        B, T, L, num_elem = species.shape

        mask = batch["mask"]
        # h_loss_mask = batch["h_mask"]
        v_loss_mask = batch["v_mask"]
        # if self.args.design:
        #     loss_mask = torch.cat([h_loss_mask, v_loss_mask], -1)
        # else:
        #     loss_mask = v_loss_mask


        B, T, L, _ = latents.shape
        assert _ == 3, f"latents shape should be (B, T, D, 3), but got {latents.shape}"
        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=species.device)
        if self.args.sim_condition:
            # x_cond = x_now[:,0]
            # species_cond = species[:,0].unsqueeze(1)
            # cell_cond = batch["cell"][:,0]
            # num_atoms_cond = batch["num_atoms"][:,0]
            # cond_mask = (mask != 0)[:,0]
            cond_mask[:, 0] = 1
        if self.args.cond_interval:
            # x_cond = x_now[:, ::self.args.cond_interval]
            # species_cond = species[:, ::self.args.cond_interval]
            # cell_cond = batch["cell"][:,::self.args.cond_interval]
            # num_atoms_cond = batch["num_atoms"][:, ::self.args.cond_interval]
            # cond_mask = (mask != 0)[:,::self.args.cond_interval]
            cond_mask[:, ::self.args.cond_interval] = 1
        cond_mask = (cond_mask*(mask!=0)) # only keep the AND set of cond_mask and mask
        if self.args.potential_model:
            return {
                "species": species,
                "latents": latents,
                'loss_mask': v_loss_mask,
                "E": batch["e_mace"],
                'model_kwargs': {
                    "aatype": species,
                    "cell": batch["cell"],
                    "num_atoms": batch["num_atoms"],
                    "conditions": None
                }
            }
        elif self.args.sim_condition or self.args.cond_interval:
            return {
                "species": batch['species_next'],
                "latents": batch['x_next'],
                'loss_mask': v_loss_mask,
                'model_kwargs': {
                    "x1": batch['x_next'],
                    'v_mask': (v_loss_mask!=0).to(int),
                    "aatype": batch['species_next'],
                    "cell": batch["cell"],
                    "num_atoms": batch["num_atoms"],
                    "conditions": {
                        'x':torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                        "mask": cond_mask,
                    }
                }
            }
        else:
            return {
                "species": species,
                "latents": latents,
                'loss_mask': v_loss_mask,
                "E": batch["e_now"],
                'model_kwargs': {
                    "x1": latents,
                    'v_mask': (v_loss_mask!=0).to(int),
                    "aatype": species,
                    "cell": batch["cell"],
                    "num_atoms": batch["num_atoms"],
                    "conditions": None
                }
            }
    
    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        start1 = time.time()
        prep = self.prep_batch(batch)

        start = time.time()
        if self.args.potential_model:
            B,T,L,_ = prep["latents"].shape
            t = torch.ones((B,), device=prep["latents"].device)
            energy = self.model(prep['latents'], t, **prep["model_kwargs"])
            energy = energy.sum(dim=2).squeeze(-1)
            # forces = -torch.autograd.grad(energy, prep['latents'])[0]
            loss_energy = (energy - prep["E"])**2
            loss = loss_energy
            self.log('loss', loss)
        else:
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


    def inference(self, batch):

        prep = self.prep_batch(batch)

        latents = prep['latents']
        B, T, N, D = latents.shape

        if self.args.design:
            # zs_continuous = torch.randn(B, T, N, self.latent_dim - 5, device=latents.device)
            zs_discrete = torch.distributions.Dirichlet(torch.ones(B, N, 5, device=latents.device)).sample()
            zs_discrete = zs_discrete[:, None].expand(-1, T, -1, -1)
            # zs = torch.cat([zs_continuous, zs_discrete], -1)
            zs = zs_discrete

            x1 = prep['latents']
            x_d = torch.zeros(x1.shape[0], x1.shape[1], x1.shape[2], 20, device=self.device)
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
            # vector_out = samples[..., :-5]
            vector_out = prep["model_kwargs"]["x_now"]
            logits = samples[..., -5:]
        else:
            vector_out = samples*prep["loss_mask"] + prep["latents"]*(1-prep["loss_mask"])

        if self.args.design:
            aa_out = torch.argmax(logits, -1)
            # aa_out = logits
        else:
            aa_out = torch.argmax(batch['species'], -1)
            # aa_out = batch['species']
        return vector_out, aa_out