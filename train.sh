#!/bin/bash
#SBATCH -t 7-0
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH -J fexs6
#SBATCH --output=fexs6.out

source activate DL
python -u Trainer.py 
