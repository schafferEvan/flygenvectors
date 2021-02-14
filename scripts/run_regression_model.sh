#!/bin/bash
#SBATCH -A axs
#SBATCH -J s0
#SBATCH -t 11:59:00
#SBATCH -c 12
#SBATCH -o r0.out -e r0.err
#SBATCH --mem-per-cpu=8gb

expID=0
module load anaconda/3-5.3.1
source activate flygenvectors
#python run_regression_model_moto.py $expID
python run_reg_new_noDown.py










