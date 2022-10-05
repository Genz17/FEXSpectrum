#!/bin/bash
#SBATCH -t 7-0
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH -J fexs
#SBATCH --output=out/fexs.out

source activate DL
python -u Trainer.py 
