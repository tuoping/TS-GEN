from mdgen.parsing import parse_train_args
args = parse_train_args()
args.pbc = True
from mdgen.logger import get_logger
logger = get_logger(__name__)

import torch, os
from mdgen.dataset import EquivariantTransformerDataset_CrCoNi
from mdgen.equivariant_wrapper import EquivariantMDGenWrapper
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import pytorch_lightning as pl

class ResetLrCallback(pl.Callback):
    def __init__(self, new_lr: float):
        self.new_lr = new_lr

    # runs right after checkpoint restore, before the first batch
    def on_train_epoch_start(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            for pg in optimizer.param_groups:
                pg["lr"] = self.new_lr
        ## (optional) reset schedulers if you wish
        # scheduler = pl_module.lr_schedulers()
        # scheduler.base_lrs = [self.new_lr]
        # scheduler.last_epoch = 1499  # starts fresh


torch.set_float32_matmul_precision('medium')

trainset = EquivariantTransformerDataset_CrCoNi(traj_dirname=args.data_dir, cutoff=args.cutoff, num_frames=args.num_frames, localmask=args.localmask, sim_condition=args.sim_condition, stage="train")

if args.overfit:
    valset = trainset    
else:
    valset = EquivariantTransformerDataset_CrCoNi(traj_dirname=args.data_dir, cutoff=args.cutoff, num_frames=args.num_frames, localmask=args.localmask, sim_condition=args.sim_condition, stage="val")

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)


# model = EquivariantMDGenWrapper(args)
checkpoint = torch.load(args.ckpt, weights_only=False)
model = EquivariantMDGenWrapper(**checkpoint["hyper_parameters"])
model.load_state_dict(checkpoint["state_dict"], strict=False)

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=0.0 if args.no_validate else (args.val_batches or 1.0),
    num_sanity_val_steps=0,
    precision=args.precision,
    enable_progress_bar=not args.wandb or os.getlogin() == 'hstark',
    gradient_clip_val=args.grad_clip,
    default_root_dir=os.environ["MODEL_DIR"], 
    callbacks=[
        ResetLrCallback(args.lr),
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"], 
            save_top_k=1,
            every_n_epochs=args.ckpt_freq,
        ),
        ModelSummary(max_depth=2),
        
    ],
    accumulate_grad_batches=args.accumulate_grad,
    val_check_interval=args.val_freq,
    check_val_every_n_epoch=args.val_epoch_freq,
    logger=False
)

# torch.manual_seed(137)
# np.random.seed(137)


if args.validate:
    # trainer.validate(model, val_loader, ckpt_path=args.ckpt)
    trainer.validate(model, val_loader)
else:
    # trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
    trainer.fit(model, train_loader, val_loader)