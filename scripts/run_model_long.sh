#!/bin/bash
#SBATCH -A axs
#SBATCH -J s3
#SBATCH -t 23:59:00
#SBATCH -c 6
#SBATCH -o r3.out -e r3.err
#SBATCH --mem-per-cpu=8gb

expID=3
module load anaconda/3-5.3.1
source activate flygenvectors
python run_reg_model_extended.py $expID











