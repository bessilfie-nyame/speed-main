#!/bin/bash
#SBATCH  --output=///srv/beegfs02/scratch/aegis_guardian/data/datasets/finetuning/training/bespeed/speed-main/logs/valn%j.out
#SBATCH  --job-name=valnd



source /itet-stor/zahmad/net_scratch/conda/etc/profile.d/conda.sh
conda activate openmmlab

python /srv/beegfs02/scratch/aegis_guardian/data/datasets/finetuning/training/bespeed/speed-main/utils.py

"$@"