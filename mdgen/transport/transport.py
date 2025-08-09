# https://github.com/willisma/SiT/
import copy

import torch
import torch as th
import numpy as np

import enum

from . import path


def mean_flat(x, mask):
    """
    Take the mean over all non-batch dimensions.
    """
    mask = mask.expand(x.shape)
    return th.sum(x * mask, dim=list(range(1, len(x.size())))) / th.sum(mask, dim=list(range(1, len(x.size()))))


from .integrators import ode, sde


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def t_to_alpha(t, args):
    """
    Convert t to alpha for Dirichlet distribution.
    """

    return 1 * (1 - t) + t * args.alpha_max, (args.alpha_max - 1)


def divergence(v_func, x, t, model_kwarg):
    # v_func: function that outputs v(x,t)
    x.requires_grad_(True)
    v = v_func(x, t, **model_kwarg)
    div = 0.0
    for i in range(x.shape[-1]):  # iterate over dimensions
        div += torch.autograd.grad(v[..., i].sum(), x, create_graph=True)[0][..., i]
    return div 


class Transport:

    def __init__(
            self,
            *,
            args,
            model_type,
            path_type,
            loss_type,
            train_eps,
            sample_eps,
            score_model = None
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }
        self.args = args
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.score_model = score_model

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)

    def check_interval(
            self,
            train_eps,
            sample_eps,
            *,
            diffusion_form="SBDM",
            sde=False,
            reverse=False,
            eval=False,
            last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
                and (
                self.model_type != ModelType.VELOCITY or sde):  # avoid numerical issue by taking a first
            # semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        x0 = []
        for i in range(5):
            x0.append(th.randn_like(x1)*self.args.x0std)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    def training_losses(
            self,
            model,
            x1,           # target tokens
            aatype1=None, # target aatype
            mask=None,
            num_species=5,
            model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """

        if model_kwargs == None:
            model_kwargs = {}
        
        ### normal sampler of t
        t, x0, x1 = self.sample(x1)
        if self.args.design:  # alterations made to the original SIT code to include dirichlet flow matching for design
            assert self.model_type == ModelType.VELOCITY
            seq_one_hot = aatype1
            ### exponential sampler of t
            # exponential_dist = torch.distributions.Exponential(1.0)
            # t = exponential_dist.sample((seq_one_hot.shape[0],)).to(seq_one_hot.device).float()
            alphas, _ = t_to_alpha(t, self.args)
            alphas = torch.ones_like(seq_one_hot) + seq_one_hot * (alphas[:, None, None, None] - torch.ones_like(seq_one_hot))
            x_d = th.distributions.Dirichlet(alphas).sample()
            xt = x_d

            # model_output = model(xt, t, cell=model_kwargs["cell"], num_atoms=model_kwargs["num_atoms"], x_cond=model_kwargs["x_cond"], x_cond_mask=model_kwargs["x_cond_mask"])
        else:
            if self.score_model is None:
                t, xt, ut = self.path_sampler.plan(t, x0[0], x1)
                if self.args.weight_loss_var_x0 > 0:
                    xt_samples = []
                    for x0_2 in x0[1:]:
                        t_2, xt_2, ut_2 = self.path_sampler.plan(t, x0_2, x1)
                        assert torch.all(t == t_2)
                        xt_samples.append(xt_2)
            else:
                t, xt, ut, st = self.path_sampler.plan_schrodinger_bridge(t, x0, x1, 3)

            '''
            ## add latent noise using antithetic sampling
            # gamma_t = (0.9*t*(1-t)).sqrt()[:,None,None,None]
            # dt_gamma_t = (0.9/2*(1-2*t)/((t*(1-t)).sqrt()))[:,None,None,None]
            # xt += gamma_t*x0
            # xt_ = xt.clone()
            # xt_ -= gamma_t*x0
            # ut += dt_gamma_t*x0
            # ut_ = ut.clone()
            # ut_ -= dt_gamma_t*x0
            '''
        
        B = x1.shape[0]
        assert t.shape == (B,)
        model_output = model(xt, t, **model_kwargs)
        if self.args.weight_loss_var_x0 > 0:
            model_output_samples = [model_output]
            for xt_2 in xt_samples:
                model_output_2 = model(xt_2, t, **model_kwargs)
                model_output_samples.append(model_output_2)
        if self.score_model is not None:
            score_model_output = self.score_model(xt, t, **model_kwargs)

            
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        if self.args.design:
            logits = model_output[:, :, :, -num_species:]
            model_output = model_output[:, :, :, :-num_species]

        terms = {}
        terms['t'] = t
        terms['pred'] = model_output
        terms['x0'] = x0
        if not (self.args.design):
            if self.model_type == ModelType.VELOCITY:
                terms["loss_continuous"]=((0.5*(model_output)**2 - (ut)*model_output))
                if self.args.weight_loss_var_x0 > 0:
                    ## model_output_samples: list of tensors, each [B, ...] same shape
                    stacked = torch.stack(model_output_samples, dim=0)  # [K, B, ...]
                    '''
                    eps = 1e-8
                    stacked_n = stacked / (stacked.norm(dim=-1, keepdim=True) + eps)
                    model_output_mean = stacked_n.mean(0, keepdim=True).detach()
                    model_output_mean = model_output_mean / (model_output_mean.norm(dim=-1, keepdim=True) + eps)
                    cos = th.nn.functional.cosine_similarity(stacked_n, model_output_mean, dim=-1)
                    terms['loss_var'] = (1-cos).mean(dim = 0)
                    '''
                    K = stacked.shape[0]
                    idx_i, idx_j = torch.triu_indices(K, K, offset=1)
                    stacked_i = stacked[idx_i]
                    stacked_j = stacked[idx_j]
                    pair_cos = torch.nn.functional.cosine_similarity(stacked_i, stacked_j, dim=-1)  
                    terms['loss_var'] = (1-pair_cos).mean(dim = 0)

                # s_est = self.path_sampler.get_score_from_velocity(model_output, xt, t)
                # div_v = divergence(model, xt, t, model_kwargs).unsqueeze(-1)
                # terms["loss_fisherreg"] = mean_flat((div_v + (model_output*s_est).sum(dim=-1).unsqueeze(-1))**2, mask)

                terms['loss_flow'] = mean_flat((0.5*(model_output)**2 - (ut)*model_output), mask)
                if self.score_model is not None:
                    terms['loss_score'] = mean_flat((0.5*(score_model_output)**2 - (st)*score_model_output), mask)
                    terms['loss'] = terms['loss_flow']+terms['loss_score']
                else:
                    if self.args.weight_loss_var_x0 > 0:
                        terms['loss'] = terms['loss_flow'] + mean_flat(terms['loss_var'], mask[...,0])*self.args.weight_loss_var_x0
                    else:
                        terms['loss'] = terms['loss_flow']
            else:
                _, drift_var = self.path_sampler.compute_drift(xt, t)
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
                if self.loss_type in [WeightType.VELOCITY]:
                    weight = (drift_var / sigma_t) ** 2
                elif self.loss_type in [WeightType.LIKELIHOOD]:
                    weight = drift_var / (sigma_t ** 2)
                elif self.loss_type in [WeightType.NONE]:
                    weight = 1
                else:
                    raise NotImplementedError()

                if self.model_type == ModelType.NOISE:
                    terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2), mask)
                else:
                    terms["loss_continuous"]=(weight * ((model_output * sigma_t + x0) ** 2)*mask)
                    terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2), mask) # loss by comparing the x_0

        # more changes for dirichlet flow matching

        if self.args.design:
            terms['loss_continuous'] = torch.tensor(torch.nan, device=xt.device)
            loss_d = th.nn.functional.cross_entropy(logits.reshape(-1,num_species), aatype1.reshape(-1,num_species).argmax(dim=-1), reduction="none").reshape(x1.shape[:-1])
            terms['loss'] = mean_flat(loss_d, mask)
            terms['loss_discrete'] = loss_d
            terms['logits'] = logits

        return terms

    def sample_latt(self, x1, cell):
        """Sampling x0 & t based on shape of x1, and the particle density.
        And reorder x0 by Hungarian algorithm over the distance matrix between x0 and x1
          Args:
            x1 - data point; [batch, *dim]
        """
        frac_x0 = (th.randn_like(x1)/2) % 1 - 0.5
        _x0 = frac_x0@cell 
        # Reorder x0 by Hungarian algorithm
        B,T,N,_ = x1.shape
        x0 = th.zeros_like(_x0)
        # for i in range(B):
        for j in range(T):
            dist_mat = (x1[:,j].unsqueeze(2)-_x0[:,j].unsqueeze(1)).norm(dim=-1)
            assignment = batch_linear_assignment(dist_mat)
            for i in range(B):
              x0[i,j] = _x0[i,j,assignment[i]]
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)

        length_prior_cell = torch.pow((cell[:,:,0,:]*torch.cross(cell[:,:,1,:], cell[:,:,2,:], dim=-1)).sum(dim=-1), 1./3.)
        cell_0 = (torch.eye(3,3).unsqueeze(0).expand(T,-1,-1).unsqueeze(0).expand(B,-1,-1,-1).to(x1.device))*length_prior_cell[:,:,None,None]

        return t, x0, x1, cell_0

    def get_drift(
            self
    ):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output)  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(
            self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        
        def score_sde(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output
        
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **model_kwargs: model(x, t, **model_kwargs) / - \
                self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **model_kwargs: model(x, t, **model_kwargs)
        elif self.model_type == ModelType.VELOCITY:
            if self.score_model is None:
                score_fn = lambda x, t, model, **model_kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **model_kwargs), x, t)
            else:
                score_fn = score_sde
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
            self,
            transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
            self,
            *,
            diffusion_form="SBDM",
            diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
            self,
            sde_drift,
            *,
            last_step,
            last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t  # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model,
                                                                                             **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
            self,
            *,
            sampling_method="Euler",
            diffusion_form="SBDM",
            diffusion_norm=1.0,
            last_step="Mean",
            last_step_size=0.04,
            num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
            self,
            *,
            sampling_method="dopri5",
            num_steps=50,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        return _ode.sample

    def sample_ode_likelihood(
            self,
            *,
            sampling_method="dopri5",
            num_steps=50,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
    ):

        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device, requires_grad=True) * 2 - 1
            if reverse:
                t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                # x.requires_grad = True
                assert x.requires_grad
                ### This way doesn't accumulate the gradient through the ODE steps
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                ### This way accumulates the gradient through the ODE steps
                # l = th.sum(self.drift(x, t, model, **model_kwargs) * eps)
                # l.backward(retain_graph=True)
                # grad = x.grad.clone()
                # x.grad.zero_()
                logp_grad = th.sum(grad * eps, dim=tuple(range(2, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size()[:2]).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            if reverse:
                logp =  delta_logp
            else:
                logp =  delta_logp
            return logp, drift

        return _sample_fn


def create_transport(
        args,
        path_type='Linear',
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        score_model=None,
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Schrodinger_Linear": PathType.LINEAR,
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]
    if (path_type in [PathType.VP]):
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0 if train_eps is None else train_eps
        sample_eps = 0 if sample_eps is None else sample_eps

    # create flow state
    state = Transport(
        args=args,
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        score_model=score_model
    )

    return state
