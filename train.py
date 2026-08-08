import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    # [Kim] Parallel data loading @ utils/data/load_data.py
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers') # Added
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=3e-4, help='Peak learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_sdum', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')

    # SDUM size (per acc4/acc8 sub-network) ## important hyperparameters
    parser.add_argument('--cascade', type=int, default=4, help='Number of SDUM cascades (unrolled iterations)')
    parser.add_argument('--channels', type=int, nargs='+', default=[24, 48, 96],
    help='Restormer channels per level (must double per level)')
    parser.add_argument('--num-blocks', type=int, nargs='+', default=[2, 3, 3],
    help='Transformer blocks per Restormer level')
    parser.add_argument('--num-heads', type=int, nargs='+', default=[1, 2, 4],
    help='MDTA attention heads per Restormer level')
    parser.add_argument('--num-refinement', type=int, default=2, help='Refinement transformer blocks')
    parser.add_argument('--mlp-ratio', type=float, default=2.0, help='GDFN expansion factor')
    parser.add_argument('--drop-path', type=float, default=0.0, help='Stochastic depth rate')
    parser.add_argument('--csm-chans', type=int, default=8, help='Channels of the coil-sensitivity U-Net')
    parser.add_argument('--csm-pools', type=int, default=4, help='Pooling layers of the coil-sensitivity U-Net')
    parser.add_argument('--no-dc-weight-map', action='store_true',
    help='Use a scalar data-consistency weight instead of the learned per-pixel map')

    # optimizer / LR schedule
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Final LR of the cosine schedule')
    parser.add_argument('--warmup-epochs', type=float, default=2.0, help='Linear warmup length in (fractional) epochs')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW weight decay')
    parser.add_argument('--lr-schedule', choices=['cosine', 'constant'], default='cosine',
    help='LR schedule after warmup')

    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--no-resume', action='store_true',
    help='Do not load the best checkpoint from a previous run; train from scratch')

    parser.add_argument('--init-from', type=Path, default=None,
    help='Checkpoint to initialize from: an AccSDUM checkpoint (loaded as-is) or a single-SDUM state_dict (loaded into BOTH acc4/acc8 sub-networks)')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of micro-batches to accumulate gradients over before each optimizer step')
    parser.add_argument('--grad-checkpointing', choices=['off', 'reentrant', 'nonreentrant'], default='nonreentrant',
    help='Gradient checkpointing of SDUM cascades: off, reentrant, or nonreentrant (default)')

    args = parser.parse_args()
    if not (len(args.channels) == len(args.num_blocks) == len(args.num_heads)):
        parser.error('--channels, --num-blocks and --num-heads must have the same number of levels')
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
