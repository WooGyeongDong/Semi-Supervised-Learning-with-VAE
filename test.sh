#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1
##
#SBATCH --job-name=Test
#SBATCH --output output/SLURM.%j%N.out
#SBATCH --error output/SLURM.%j%N.err
##
#SBATCH --gres=gpu:rtx3090:1


module add CUDA/11.3.0

python VAE_Classifier.py