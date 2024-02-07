#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu
#SBATCH --qos=long
#SBATCH --time=24:00:00
#SBATCH --partition=aa100

module purge
module load anaconda

conda activate kgenv

python extract_kg_triples.py hotpot_2ndhalfvalid.json