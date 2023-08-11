#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --gpus-per-node=t4:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00

cd /home/guest185/SPARK_Stater/BRATS2023_final/
module load python/3.8
source /home/guest185/hackathon_38/bin/activate
python main.py \
    --json_list='./brats2023_africa_data.json' \
    --data_dir='/scratch/guest185/BraTS_Africa_data/' \
    --max_epochs=100 \
    --save_checkpoint \
    --distributed \
    --optim_name='adadelta' \
    --optim_lr=1.0 \
    --lrschedule='step_lr' \
    --optim_step_size=1 \
    --optim_gamma=0.97 \
    --sw_batch_size=4 \
    --batch_size=1 \
    --val_every=50 \
    --infer_overlap=0.7 \
    --in_channels=4 \
    --spatial_dims=3 \
    --use_checkpoint \
    --feature_size=60 \
    --logdir='4_gpu_60_epochs'

