#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --gpus-per-node=t4:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=25:00:00

cd /home/guest185/SPARK_Stater/Africa_SSA_Model/
module load python/3.8
source /home/guest185/hackathon_38/bin/activate

python ssa_swin_model.py