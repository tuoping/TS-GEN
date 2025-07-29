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
            num_scalar_out = self.args.num_species
            num_vector_out=0
            latent_dim = self.args.num_species
        else:
            num_scalar_out = 0
            num_vector_out=1
            latent_dim = 3
        
        encoder = Encoder_dpm(num_species, args.embed_dim, 4, args.edge_dim, input_dim=1)
        processor = Processor(num_convs=args.num_convs, node_dim=args.embed_dim, num_heads=args.num_heads, ff_dim=args.ff_dim, edge_dim=args.edge_dim)
        print("Initializing drift model")
        self.model = EquivariantTransformer_dpm(
            encoder = encoder,
            processor = processor,
            decoder = Decoder(dim=args.embed_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out, num_species=args.num_species),
            cutoff=args.cutoff,
            latent_dim=latent_dim,
            embed_dim=args.embed_dim,
            design=args.design,
            potential_model = False,
            tps_condition=args.tps_condition,
            num_species=args.num_species,
            pbc=args.pbc,
        )
        if args.potential_model:
            num_scalar_out = 1
            latent_dim = 3
            num_vector_out = 0
            edge_dim = 64
            embed_dim = 4
            self.potential_model = EquivariantTransformer_dpm(
                encoder = Encoder_dpm(num_species, embed_dim, 4, edge_dim, input_dim=1),
                processor = Processor(num_convs=args.num_convs, node_dim=embed_dim, num_heads=args.num_heads, ff_dim=args.ff_dim, edge_dim=edge_dim),
                decoder = Decoder(dim=embed_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out),
                cutoff=args.cutoff,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                design=args.design,
                potential_model = args.potential_model,
                tps_condition=args.tps_condition,
                num_species=args.num_species,
                pbc=args.pbc,
            )
            # for name, p in self.potential_model.named_parameters():
            #     print(f"{name}: {p.requires_grad}")
            # for param in self.potential_model.parameters():
            #     param.requires_grad = False
        if args.path_type == "Schrodinger_Linear":
            print("Initializing score model")
            self.score_model = EquivariantTransformer_dpm(
                encoder = encoder,
                processor = processor,
                decoder = Decoder(dim=args.embed_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out),
                cutoff=args.cutoff,
                latent_dim=latent_dim,
                embed_dim=args.embed_dim,
                design=args.design,
                potential_model = args.potential_model,
                tps_condition=args.tps_condition,
                pbc=args.pbc,
            )
        else:
            self.score_model = None

        self.transport = create_transport(
            args,
            args.path_type,
            args.prediction,
            train_eps=1e-5,
            sample_eps=1e-5,
            score_model=self.score_model
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
        species = batch["species"]
        latents = batch["x"]
        # rdf = batch["RDF"]
        B, T, L, num_elem = species.shape

        v_loss_mask = batch["v_mask"]


        B, T, L, _ = latents.shape
        assert _ == 3, f"latents shape should be (B, T, D, 3), but got {latents.shape}"
        ########
        
        if "TKS_mask" not in batch.keys():
            batch['TKS_mask'] = torch.ones(B,T,L, dtype=int, device=species.device)
            batch['TKS_v_mask'] = torch.ones(B,T,L,3, dtype=int, device=species.device)

        if self.args.sim_condition:
            cond_mask = torch.zeros(B, T, L, dtype=int, device=species.device)
            cond_mask[:, 0] = 1
            if self.stage == "inference":
                conditional_batch = True
            else:
                # conditional_batch = torch.rand(1)[0] >= 0.7
                conditional_batch = True
        elif self.args.tps_condition:
            cond_mask_f = torch.zeros(B, T, L, dtype=int, device=species.device)
            cond_mask_r = torch.zeros(B, T, L, dtype=int, device=species.device)
            cond_mask = torch.zeros(B, T, L, dtype=int, device=species.device)
            cond_mask_f[:, 0] = 1
            cond_mask_r[:, -1] = 1
            cond_mask[:, 1:-1] = 1
            if self.stage == "inference":
                conditional_batch = True
            else:
                # conditional_batch = torch.rand(1)[0] >= 0.7
                conditional_batch = True

        if (self.args.sim_condition and conditional_batch):
            # For sim_condition, the x and x_next are separately feeded.
            if self.args.potential_model:
                return [{
                        "species": batch['species_next'],
                        "latents": batch['x_next'],
                        'loss_mask': batch["TKS_v_mask"],
                        'model_kwargs': {
                            "x1": batch['x_next'],
                            'v_mask': (batch["TKS_v_mask"]!=0).to(int),
                            "aatype": batch['species_next'],
                            "cell": batch['cell'],
                            "num_atoms": batch["num_atoms"],
                            "conditions": {
                                'x': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                                "mask": cond_mask*(batch["TKS_mask"]!=0),
                                'cell': batch['cell'],
                                'species': batch['species'],
                                'num_atoms': batch['num_atoms'],
                                'model': self.potential_model
                            }
                        }
                },
                        {
                        "species": species,
                        "latents": latents,
                        'E': batch['e_now'],
                        'model_kwargs': {
                            "aatype": species,
                            'x1': latents,
                            'v_mask': torch.zeros_like(v_loss_mask, dtype=int, device=species.device),
                            "cell": batch['cell'],
                            "num_atoms": batch["num_atoms"],
                            "conditions": None
                        }
                }]
            else:
                return {
                    "species": batch['species_next'],
                    "latents": batch['x_next'],
                    'loss_mask': batch["TKS_v_mask"],
                    'model_kwargs': {
                        "x1": batch['x_next'],
                        'v_mask': (batch["TKS_v_mask"]!=0).to(int),
                        "aatype": batch['species_next'],
                        "cell": batch['cell'],
                        "num_atoms": batch["num_atoms"],
                        "conditions": {
                            'x':torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                            "mask": cond_mask,
                            'cell': batch['cell'],
                            'species': batch['species'],
                            'num_atoms': batch['num_atoms']
                        }
                    }
                }
        elif (self.args.tps_condition and conditional_batch):
            # For tps_condition, the x[:::] are feeded together, v_mask is not necessary.
            return {
                "species": species,
                "latents": latents,
                'E': batch['e_now'],
                'loss_mask': batch["TKS_v_mask"]*cond_mask.unsqueeze(-1),
                'loss_mask_potential_model': (batch["TKS_mask"]!=0).to(int)[:,:,0]*cond_mask[:,:,0],
                'model_kwargs': {
                    "x1": latents,
                    'v_mask': (batch["TKS_v_mask"]!=0).to(int)*cond_mask.unsqueeze(-1),
                    "aatype": species,
                    "cell": batch['cell'],
                    "num_atoms": batch["num_atoms"],
                    "conditions": {
                        'cond_f':{
                            'x': torch.where(cond_mask_f.unsqueeze(-1).bool(), latents, 0.0).reshape(-1,3),
                            'mask': (batch["TKS_v_mask"]!=0).to(int)*cond_mask_f.reshape(-1),
                        },
                        'cond_r':{
                            'x': torch.where(cond_mask_r.unsqueeze(-1).bool(), latents, 0.0).reshape(-1,3),
                            'mask': (batch["TKS_v_mask"]!=0).to(int)*cond_mask_r.reshape(-1),
                        }
                    }
                }
            }
        else:
            return {
                "species": species,
                "latents": latents,
                'loss_mask': v_loss_mask,
                'model_kwargs': {
                    "aatype": species,
                    'x1': latents,
                    'v_mask': (v_loss_mask!=0).to(int),
                    "cell": batch['cell'],
                    "num_atoms": batch["num_atoms"],
                    "conditions": None
                }
            }
    
    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        start1 = time.time()
        if self.args.sim_condition and self.args.potential_model:
            prep, prep_potential_model = self.prep_batch(batch)
        elif self.args.tps_condition:
            prep = self.prep_batch(batch)
            prep_potential_model = prep
        else:
            prep = self.prep_batch(batch)

        start = time.time()
        loss = 0.
        
        out_dict = self.transport.training_losses(
            model=self.model,
            x1=prep['latents'],
            aatype1=batch['species'],
            mask=prep['loss_mask'],
            model_kwargs=prep['model_kwargs']
        )
        self.log('model_dur', time.time() - start)
        self.log('time', out_dict['t'])
        loss_gen = out_dict['loss']
        self.log('loss_gen', loss_gen)
        loss = loss_gen
        if self.score_model is not None:
            self.log("loss_flow", out_dict['loss_flow'])
            self.log("loss_score", out_dict['loss_score'])

        if self.args.potential_model:
            B,T,L,_ = prep_potential_model["latents"].shape
            t = torch.ones((B,), device=prep_potential_model["latents"].device)
            energy = self.potential_model(prep_potential_model['latents'], t, **prep_potential_model["model_kwargs"])
            energy = energy.sum(dim=2).squeeze(-1)
            # forces = -torch.autograd.grad(energy, prep_potential_model['latents'])[0]
            loss_energy = ((energy -prep_potential_model["E"])**2).sum(-1).squeeze(-1)
            self.log('loss_energy', loss_energy)  
            self.log('energy', prep_potential_model["E"]) 
            self.log('pred_energy', energy) 
            loss += loss_energy

        self.log('model_dur', time.time() - start)
        self.log('loss', loss)

        self.log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.log('name', ','.join(batch['name']))
        self.log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()
        
        return loss.mean()

    def guided_velocity(self, x, t, cell=None, 
                num_atoms=None,
                conditions=None,
                aatype=None, x1=None, v_mask=None):
        with torch.no_grad(): 
            v = self.model.forward_inference(x, t,                 
                cell=cell, 
                num_atoms=num_atoms,
                conditions=conditions,
                aatype=aatype, x1=x1, v_mask=v_mask)
        B,T,L,_ = x.shape
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = -torch.autograd.grad(self.potential_model(x, torch.ones((B,), device=x.device).detach().requires_grad_(False), 
                    cell=cell, 
                    num_atoms=num_atoms,
                    conditions=conditions,
                    aatype=aatype, x1=x1, v_mask=v_mask).sum(dim=2).squeeze(-1)[:,1], x, create_graph=False)[0].detach()
        self.integration_step += 1
        return v + self.args.guidance_pref*g

    def inference(self, batch, stage='inference'):
        self.stage = stage
        if self.args.sim_condition and self.args.potential_model:
            prep, prep_potential_model = self.prep_batch(batch)
        elif self.args.tps_condition:
            prep = self.prep_batch(batch)
            prep_potential_model = prep
        else:
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

        if self.score_model is None:
            sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method)  # default to ode
        else:
            sample_fn = self.transport_sampler.sample_sde(num_steps=self.args.inference_steps, diffusion_form=self.args.diffusion_form, diffusion_norm=torch.tensor(3))

        with torch.no_grad():
            samples = sample_fn(
                zs,
                partial(self.model.forward_inference, **prep['model_kwargs'])
            )[-1]
        
        if self.args.design:
            # vector_out = samples[..., :-self.args.num_species]
            vector_out = prep["model_kwargs"]["x_now"]
            logits = samples[..., -self.args.num_species:]
        else:
            vector_out = samples*prep["loss_mask"] + prep["latents"]*(1-prep["loss_mask"])

        if self.args.design:
            aa_out = torch.argmax(logits, -1)
            # aa_out = logits
        else:
            aa_out = torch.argmax(batch['species'], -1)
            # aa_out = batch['species']
        return vector_out, aa_out
    
