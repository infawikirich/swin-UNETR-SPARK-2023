#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --gpus-per-node=t4:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=40:00:00

cd /home/guest185/SPARK_Stater/BRATS2023/
module load python/3.8
source /home/guest185/hackathon_38/bin/activate
# python swin_model.py
# python kfold_json_generator_final.py
# python SGDWarmRestart.py
python RAdamLR.py