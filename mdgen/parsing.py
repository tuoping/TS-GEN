from argparse import ArgumentParser
import os


def parse_train_args():
    parser = ArgumentParser()

    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    
    ## Epoch settings
    group = parser.add_argument_group("Epoch settings")
    group.add_argument("--epochs", type=int, default=1000)
    group.add_argument("--overfit", action='store_true')
    group.add_argument("--overfit_peptide", type=str, default=None)
    group.add_argument("--overfit_frame", action='store_true')
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--val_repeat", type=int, default=25)
    group.add_argument("--inference_batches", type=int, default=0)
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--val_freq", type=float, default=None)
    group.add_argument("--val_epoch_freq", type=int, default=1)
    group.add_argument("--no_validate", action='store_true')
    group.add_argument("--designability_freq", type=int, default=1)

    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=40)
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--run_name", type=str, default="default")
    

    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=1)
    group.add_argument("--grad_clip", type=float, default=1.)
    group.add_argument("--check_grad", action='store_true')
    group.add_argument('--grad_checkpointing', action='store_true')
    group.add_argument('--adamW', action='store_true')
    group.add_argument('--ema', action='store_true')
    group.add_argument('--ema_decay', type=float, default=0.999)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument('--precision', type=str, default='32-true')
    
    ## Training data 
    group = parser.add_argument_group("Training data settings")
    group.add_argument('--data_dir', type=str, default=None, required=True)
    group.add_argument('--num_frames', type=int, default=1)
    group.add_argument('--suffix', type=str, default='')

    ### Masking settings
    group = parser.add_argument_group("Masking settings")
    group.add_argument('--localmask', action='store_true')

    ## Ablations settings
    group = parser.add_argument_group("Ablations settings")
    
    
    ## Model settings
    group = parser.add_argument_group("Model settings")
    group.add_argument('--dropout', type=float, default=0.0)
    group.add_argument('--scale_factor', type=float, default=1.0)
    group.add_argument('--num_layers', type=int, default=5)
    group.add_argument('--embed_dim', type=int, default=384)

    group.add_argument('--time_multiplier', type=float, default=100.)
    group.add_argument('--abs_pos_emb', action='store_true')
    group.add_argument('--abs_time_emb', action='store_true')

    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="GVP", choices=["Linear", "GVP", "VP", "Schrodinger_Linear"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--sampling_method", type=str, default="dopri5", choices=["dopri5", "euler"])
    group.add_argument('--alpha_max', type=float, default=8)
    group.add_argument('--discrete_loss_weight', type=float, default=0.5)
    group.add_argument("--dirichlet_flow_temp", type=float, default=1.0)
    group.add_argument('--allow_nan_cfactor', action='store_true')
    # group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    

    ## video settings
    group = parser.add_argument_group("Video settings")
    group.add_argument('--tps_condition', action='store_true')
    group.add_argument('--design', action='store_true')
    group.add_argument('--design_from_traj', action='store_true')
    group.add_argument('--sim_condition', action='store_true')
    group.add_argument('--frame_interval', type=int, default=1)
    group.add_argument('--cond_interval', type=int, default=None) # for superresolution
    
    ## Equivariant Transformer settings
    group.add_argument('--num_species', type=int, default=5)
    group.add_argument('--edge_dim', type=int, default=64)
    group.add_argument('--num_convs', type=int, default=5)
    group.add_argument('--num_heads', type=int, default=4)
    group.add_argument('--ff_dim', type=int, default=64)
    group.add_argument('--cutoff', type=float, default=2.5)
    
    ## nonequil. simulation settings
    group.add_argument('--potential_model', action='store_true')
    group.add_argument("--pbc", action='store_true')
    group.add_argument("--guided", action='store_true')

    ## SDE bridge settings
    group.add_argument("--inference_steps", type=int, default=20)

    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)

    return args


