#!/bin/bash
#SBATCH -t 7-0
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH -J fexs4
#SBATCH --output=out/fexs4.out

source activate DL
python -u Trainer.py
