#!/bin/bash

# build
# python3 setup.py develop

# lego demo
# python3 examples/train_ngp_nerf.py --train_split train --scene lego --data_root /mnt/data/DATASET/NERF_dataset/NERF/nerf_synthetic

# cloth demo
# python3 examples/train_ngp.py --train_split train --scene cloth --data_root /mnt/data/DATASET/NERF_dataset/NERF_my/
python3 examples/show_gui.py --train_split train --scene man5 --data_root /mnt/data/DATASET/NERF_dataset/NERF_my/