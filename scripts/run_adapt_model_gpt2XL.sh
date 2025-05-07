#!/bin/bash
#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint="vram40"


# This script for running DAMA


python adapt_model.py --model_name "gpt2-xl" --method "ALPHA_EDIT" --request_file "train_dama_processed.json" --num_layers 2  --post_linear True --iterative_update True --random_seed 45
