#!/bin/bash
#SBATCH -A axs
#SBATCH -J srcx
#SBATCH -t 11:59:00
#SBATCH -c 12
#SBATCH -o r3.out -e r3.err
#SBATCH --mem-per-cpu=8gb

expID=3
module load anaconda/3-5.3.1
source activate flygenvectors
python get_map_params_.py $expID







