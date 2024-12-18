#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH --partition=aa100

module purge
module load anaconda

conda activate lcm

python run_experiment.py --config config.json