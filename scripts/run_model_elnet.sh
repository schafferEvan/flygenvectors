#!/bin/bash
#SBATCH -A axs
#SBATCH -J s9
#SBATCH -t 11:59:00
#SBATCH -c 12
#SBATCH -o r9.out -e r9.err
#SBATCH --mem-per-cpu=8gb

expID=9
module load anaconda/3-5.3.1
source activate flygenvectors
python run_reg_model_elastic_net.py $expID











