#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --reservation hackathon-wr_gpu
#SBATCH --gpus-per-node=t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00

module load python/3.9
source /home/guest185/hackathon/bin/activate
python main.py
