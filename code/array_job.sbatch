#!/bin/bash

#SBATCH --job-name=d4_10
#SBATCH --output=/n/home13/asaxe/gendynamics/results/expt4/d4_%a_10.out
#SBATCH --error=/n/home13/asaxe/gendynamics/results/expt4/d4_%a_10.err
#SBATCH -t 0-1:00
#SBATCH -p serial_requeue
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=1500

cd /Users/asaxe/Documents/postdoctoralwork/gendynamics/code
module load keras
python run_indep_gaussian_array.py -paramfile params.txt -line ${SLURM_ARRAY_TASK_ID}
