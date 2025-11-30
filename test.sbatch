#!/bin/bash
#SBATCH -J optimization
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -o logs/optim-%j.out
#SBATCH -e logs/optim-%j.err
#SBATCH -t 48:00:00
#SBATCH -p epyc-256
#SBATCH --mail-type=END

python test.py