#!/usr/bin/env bash
# Score reconstructions already written by reconstruct.sh.
DATA=${DATA:-/root/Data}
python leaderboard_eval.py \
  -lp "$DATA/leaderboard" \
  -yp '../result/test_Varnet/reconstructions_leaderboard'
