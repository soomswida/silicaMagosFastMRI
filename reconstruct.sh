#!/usr/bin/env bash
# Reconstruct the leaderboard set (acc4 + acc8) with ../result/<net_name>/checkpoints/best_model.pt.
DATA=${DATA:-/root/Data}
python reconstruct.py \
  -b 1 \
  -n 'test_Varnet' \
  -p "$DATA/leaderboard"
