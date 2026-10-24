#!/usr/bin/env bash
# Train the dual acc4/acc8 VarNet. Adjust DATA to where the challenge data lives
# (expects $DATA/train and $DATA/val, see README "Data layout").
DATA=${DATA:-/root/Data}

# Final submission configuration (full-size E2E-VarNet, both sub-networks).
# Gradient checkpointing keeps the 12-cascade model inside ~8 GB of VRAM.
python train.py \
  -b 1 \
  -e 100 \
  -l 0.001 \
  -r 10 \
  -n 'test_Varnet' \
  --cascade 12 --chans 18 --sens_chans 8 \
  --grad-checkpointing \
  -t "$DATA/train/" \
  -v "$DATA/val/"

# Smaller configuration used for quick baseline comparisons:
# python train.py -b 1 -e 5 -l 0.001 -r 10 -n 'test_Varnet_small' \
#   --cascade 3 --chans 12 --sens_chans 6 --grad-checkpointing \
#   -t "$DATA/train/" -v "$DATA/val/"
