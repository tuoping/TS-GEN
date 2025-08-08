import torch, os
from mdgen.dataset import EquivariantTransformerDataset_Transition1x

trainset = EquivariantTransformerDataset_Transition1x("tests/test_data/Transition1x/", 12, sim_condition=False, tps_condition=True, stage="train")

batch_size = 1
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
)
sample_batch = next(iter(train_loader))


from argparse import Namespace

args = Namespace(ckpt=None, validate=False, num_workers=4, epochs=10000, overfit=False, train_batches=None, val_batches=None, val_repeat=25, inference_batches=0, batch_size=128, val_freq=None, val_epoch_freq=1, no_validate=False, designability_freq=1, print_freq=100, ckpt_freq=40, wandb=False, run_name='default', accumulate_grad=1, grad_clip=1.0, check_grad=False, grad_checkpointing=False, adamW=False, ema=False, ema_decay=0.999, lr=0.0001, precision='32-true', suffix="", dropout=0.0, scale_factor=1.0, num_layers=5, embed_dim=64, time_multiplier=100.0, abs_pos_emb=True, abs_time_emb=False, prediction='velocity', sampling_method='dopri5', alpha_max=8, discrete_loss_weight=0.5, dirichlet_flow_temp=1.0, allow_nan_cfactor=False, tps_condition=True, design=False, design_from_traj=False, frame_interval=None, cond_interval=None)

args.data_dir="tests/test_data/Transition1x" 
args.num_frames=3

args.ckpt_freq = 10 
args.val_repeat = 0.2
args.epochs = 1
args.num_species = 5

args.edge_dim = 4
args.num_convs = 5
args.num_heads = 4
args.ff_dim = 16

args.cutoff= 12
args.localmask = True
args.path_type = "Linear"
args.batch_size = batch_size

args.tps_condition = True
args.sim_condition = False
# args.prediction='score'
# args.sampling_method = "Euler"

args.design = False
args.potential_model = False
args.pbc = False 

os.environ["MODEL_DIR"] = os.path.join("tests_cache", args.run_name)

from mdgen.equivariant_wrapper import EquivariantMDGenWrapper
model = EquivariantMDGenWrapper(args)

model.iter_step += 1
model.stage = "train"
prep = model.prep_batch(sample_batch)

_, x0, x1 = model.transport.sample(prep['latents'])

# Test t=0
model_output = model.model(x0, torch.tensor([0]), **prep['model_kwargs'])
x = x0*prep['model_kwargs']['v_mask']+ prep['model_kwargs']['x1']*(1-prep['model_kwargs']['v_mask'])
assert torch.allclose(x0[:,1,...], x[:,1,...])
assert not torch.allclose(x0[:,0,...], x[:,0,...])
assert not torch.allclose(x0[:,2,...], x[:,2,...])
vec_raw = model.model.inference(x, torch.tensor([0]), cell=prep['model_kwargs']['cell'], num_atoms=prep['model_kwargs']['num_atoms'], conditions=prep['model_kwargs']['conditions'], aatype=prep['model_kwargs']['aatype'])

assert torch.allclose(vec_raw[:,1,...], model_output[:,1,...])
assert not torch.allclose(vec_raw[:,0,...], model_output[:,0,...])
assert not torch.allclose(vec_raw[:,2,...], model_output[:,2,...])