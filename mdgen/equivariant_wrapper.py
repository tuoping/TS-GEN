from .ema import ExponentialMovingAverage
from .logger import get_logger

logger = get_logger(__name__)

import torch, time
from torch import nn
import copy
import numpy as np
from functools import partial

from .model.equivariant_latent_model import EquivariantTransformer_dpm, Encoder_dpm, Processor, Decoder
from .wrapper import Wrapper, gather_log, get_log_mean

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
        elif args.pbc:
            num_scalar_out = 0
            num_vector_out=1
            latent_dim = args.embed_dim
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
            self.potential_model = EquivariantTransformer_dpm(
                encoder = Encoder_dpm(num_species, args.embed_dim, 4, args.edge_dim, input_dim=1),
                processor = Processor(num_convs=args.num_convs, node_dim=args.embed_dim, num_heads=args.num_heads, ff_dim=args.ff_dim, edge_dim=args.edge_dim),
                decoder = Decoder(dim=args.embed_dim, num_scalar_out=num_scalar_out, num_vector_out=num_vector_out, num_species=args.num_species),
                cutoff=args.cutoff,
                latent_dim=latent_dim,
                embed_dim=args.embed_dim,
                design=args.design,
                potential_model = args.potential_model,
                tps_condition=args.tps_condition,
                num_species=args.num_species,
                pbc=args.pbc,
            )
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

    def on_validation_epoch_end(self):
        if self.args.ema:
            self.restore_cached_weights()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)
        self.log("val_loss", mean_log['val_loss'])
        if self.args.weight_loss_var_x0 > 0:
            self.log("val_loss_var", mean_log['val_loss_var'])
        self.log("val_loss_gen", mean_log['val_loss_gen'])
        self.log("val_meanRMSD", mean_log['val_meanRMSD'])
        self.print_log(prefix='val', save=False)

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
                conditional_batch = torch.rand(1)[0] >= 1-self.args.ratio_conditonal
                # conditional_batch = True

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
                conditional_batch = torch.rand(1)[0] >= 1-self.args.ratio_conditonal
                # conditional_batch = True

        if (self.args.sim_condition and conditional_batch):
            # For sim_condition, the x and x_next are separately feeded.
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
                        "mask": cond_mask*(batch["TKS_mask"]!=0),
                        'cell': batch['cell'],
                        'species': batch['species'],
                        'num_atoms': batch['num_atoms']
                    }
                },
                'conditional_batch': conditional_batch
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
                            'x': torch.where(cond_mask_f.unsqueeze(-1).bool(), latents, 0.0)[:,0,...].unsqueeze(1).expand(B,T,L,3).reshape(-1,3),
                            'mask': cond_mask_f[:,0,...].unsqueeze(1).expand(B,T,L).reshape(-1),
                        },
                        'cond_r':{
                            'x': torch.where(cond_mask_r.unsqueeze(-1).bool(), latents, 0.0)[:,-1,...].unsqueeze(1).expand(B,T,L,3).reshape(-1,3),
                            'mask': cond_mask_r[:,-1,...].unsqueeze(1).expand(B,T,L).reshape(-1),
                        }
                    }
                },
                'conditional_batch': conditional_batch
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
                },
                'conditional_batch': conditional_batch
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
        self.prefix_log('model_dur', time.time() - start)
        self.prefix_log('time', out_dict['t'])
        self.prefix_log('conditional_batch', prep['conditional_batch'].to(torch.float32))
        loss_gen = out_dict['loss']
        self.prefix_log('loss_gen', loss_gen)
        if self.args.weight_loss_var_x0 > 0:
            self.prefix_log('loss_var', out_dict['loss_var'])
        loss = loss_gen
        if self.score_model is not None:
            self.prefix_log("loss_flow", out_dict['loss_flow'])
            self.prefix_log("loss_score", out_dict['loss_score'])

        if self.args.potential_model:
            B,T,L,_ = prep["latents"].shape
            t = torch.ones((B,), device=prep["latents"].device)
            energy = self.potential_model(prep['latents'], t, **prep["model_kwargs"])
            energy = energy.sum(dim=2).squeeze(-1)
            # forces = -torch.autograd.grad(energy, prep['latents'])[0]
            loss_energy = (((energy -prep["E"])**2)*prep['loss_mask_potential_model']).sum(-1)
            self.prefix_log('loss_energy', loss_energy)        
            loss += loss_energy
        self.prefix_log('model_dur', time.time() - start)
        self.prefix_log('loss', loss)

        self.prefix_log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.prefix_log('name', ','.join(batch['name']))
        self.prefix_log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()
        if stage == "val":
            B,T,L,_ = prep['latents'].shape
            with torch.no_grad():
                pred_pos, _ = self.inference(batch, stage=stage)
                ref_pos = prep['latents']
                ## (\Delta d per atom) # B,T,L
                err = ((((pred_pos - ref_pos)*(prep['loss_mask']!=0)).norm(dim=-1)))
                ## RMSD per configuration # B,T
                err = ((err**2).mean(dim=-1)).sqrt()
                ## mean RMSD per sample # B
                err = err.mean(dim=-1)
                self.prefix_log('meanRMSD', err)
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
            zs = torch.randn(B, T, N, D, device=self.device)*self.args.x0std

        self.integration_step = 0
        if self.score_model is None:
            with torch.no_grad(): sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method, num_steps=self.args.inference_steps)  # default to ode
        else:
            with torch.no_grad(): sample_fn = self.transport_sampler.sample_sde(num_steps=self.args.inference_steps, diffusion_form=self.args.diffusion_form, diffusion_norm=torch.tensor(3))


        if self.args.guided:
            with torch.no_grad(): samples = sample_fn(
                    zs,
                    partial(self.guided_velocity, **prep['model_kwargs'])
                )[-1]
        else:
            samples = sample_fn(
                zs,
                partial(self.model.forward_inference, **prep['model_kwargs'])
            )[-1]
        
        if self.args.design:
            # vector_out = samples[..., :-self.args.num_species]
            vector_out = prep["model_kwargs"]["x_now"]
            logits = samples[..., -self.args.num_species:]
        else:
            vector_out = samples *prep["model_kwargs"]['v_mask'] + prep["latents"]*(1-prep["model_kwargs"]['v_mask'])

        if self.args.design:
            aa_out = torch.argmax(logits, -1)
            # aa_out = logits
        else:
            aa_out = torch.argmax(batch['species'], -1)
            # aa_out = batch['species']
        return vector_out, aa_out
    
