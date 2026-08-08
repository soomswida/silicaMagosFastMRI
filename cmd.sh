# -e 50 (Aug 7)
python train.py -b 1 -e 100 -l 0.001 -r 10 -n 'te
st_Varnet' --cascade 12 --chans 18 --sens_chans 8 -t '/root/Data/train/' -v '/root/Data/val/' --grad-c
heckpointing
