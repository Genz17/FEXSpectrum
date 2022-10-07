#!/bin/bash
#SBATCH -t 7-0
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH -J fexs4_sep
#SBATCH --output=out/fexs4_sep.out

source activate DL
python -u Trainer.py 
