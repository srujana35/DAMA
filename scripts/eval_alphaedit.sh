#!/bin/bash
#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:00:00  # Job time limit
#SBATCH -o ./jobs-exp/%j.out  # %j = job ID
#SBATCH --constraint="vram24"


ds_split="test"
ds_prefix=("anti_type1" "pro_type1" "anti_type2" "pro_type2")

python evaluate_model.py --param_number 7 --method "ALPHA_EDIT" --test_file wikitext_wikitext-103-raw-v1 --test_task "causal_lm" --model_name gpt2-xl
python evaluate_model.py --param_number 7 --method "ALPHA_EDIT" --test_file ${ds_split}_dama.json --test_task "gen" --model_name gpt2-xl

for ds in "${ds_prefix[@]}"
do
  python evaluate_model.py --param_number 7 --method "ALPHA_EDIT" --test_file "${ds}_${ds_split}.txt" --test_task "coref" --model_name gpt2-xl
done