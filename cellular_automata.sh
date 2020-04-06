#!/bin/env bash

#SBATCH --partition=shared-gpu-EL7
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

srun cellular_automata cyclic.cl 7 1 3000 3000 10000 