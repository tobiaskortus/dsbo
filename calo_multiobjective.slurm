#!/bin/bash
#SBATCH -J optimization
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o logs/optim-%j.out
#SBATCH -e logs/optim-%j.err
#SBATCH -p skylake-96
#SBATCH --mem=16GB
#SBATCH -t 48:00:00

#mprof run --multiprocess python -m memory_profiler calo_multiobjective.py
python calo_multiobjective.py