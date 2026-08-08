import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_part import forward
import time

    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')

    # Model hyperparameters are read from checkpoint['args']; these optional
    # flags override individual values (rarely needed).
    parser.add_argument('--sdum-cascade', type=int, default=None)
    parser.add_argument('--sdum-channels', type=int, nargs='+', default=None)
    parser.add_argument('--sdum-num-blocks', type=int, nargs='+', default=None)
    parser.add_argument('--sdum-num-heads', type=int, nargs='+', default=None)
    parser.add_argument('--sdum-num-refinement', type=int, default=None)
    parser.add_argument('--sdum-mlp-ratio', type=float, default=None)
    parser.add_argument('--sdum-csm-chans', type=int, default=None)
    parser.add_argument('--sdum-csm-pools', type=int, default=None)
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'

    # acc4
    args.data_path = args.path_data / "acc4"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    print(args.forward_dir)
    start_acc4 = time.time()
    forward(args)
    time_acc4 = time.time() - start_acc4

    # acc8
    args.data_path = args.path_data / "acc8"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc8"
    print(args.forward_dir)
    start_acc8 = time.time()
    forward(args)
    time_acc8 = time.time() - start_acc8

    print(f'Reconstruction Time (acc4): {time_acc4:.2f}s')
    print(f'Reconstruction Time (acc8): {time_acc8:.2f}s')
    print(f'Total Reconstruction Time : {time_acc4 + time_acc8:.2f}s')
    print('Done!')