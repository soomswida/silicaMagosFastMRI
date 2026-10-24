#!/usr/bin/env bash
# Reconstruct + score the leaderboard set in one pass (organiser's fixed harness).
DATA=${DATA:-/root/Data}
python recon_eval.py \
  -n 'test_Varnet' \
  -p "$DATA/leaderboard"
