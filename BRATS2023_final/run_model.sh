#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --gpus-per-node=t4:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00

cd /home/guest185/SPARK_Stater/BRATS2023_final/
module load python/3.8
source /home/guest185/hackathon_38/bin/activate
srun python main.py \
                --json_list='./brats2023_africa_data.json' \
                --data_dir='/scratch/guest185/BraTS_Africa_data/' \
                --max_epochs=100 \
                --save_checkpoint \
                --distributed \
                --lrschedule='cosine_anneal' \
                --sw_batch_size=2 \
                --batch_size=1 \
                --val_every=2 \
                --infer_overlap=0.8 \
                --in_channels=4 \
                --spatial_dims=3 \
                --use_checkpoint \
                --feature_size=48 \
                --logdir='4_gpu_60_epochs'
