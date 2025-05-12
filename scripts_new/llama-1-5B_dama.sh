python adapt_model.py --model_name "meta-llama/Llama-3.2-1B" --method "DAMA" --request_file "train_dama.json" --num_layers 2 --post_linear True --iterative_update True --random_seed 45

# python evaluate_model.py --model_name "meta-llama/Llama-3.2-1B" --num_layers 2 --method "MEMIT" --test_file test_dama.json --test_task gen --post_linear True --iterative_update True --random_seed 45
# python evaluate_model.py --model_name "meta-llama/Llama-3.2-1B" --num_layers 0 --method "MEMIT" --test_file "anti_type1_test.txt" --test_task coref --post_linear True --iterative_update True --random_seed 45
# python evaluate_model.py --model_name "meta-llama/Llama-3.2-1B" --num_layers 2 --method "MEMIT" --test_file stereoset_dev.json --test_task stereoset --post_linear True --iterative_update True --random_seed 45
# python evaluate_model.py --model_name "meta-llama/Llama-3.2-1B" --num_layers 2 --method "MEMIT" --test_file wikitext_wikitext-2-raw-v1 --test_task "causal_lm" --post_linear True --iterative_update True --random_seed 45
