#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --reservation hackathon-wr_gpu
#SBATCH --gpus-per-node=t4:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:01:00

cd /home/guest185/SPARK_Stater/data-exploratory
module load python/3.9
source /home/guest185/hackathon/bin/activate
python data_exploratory.py
