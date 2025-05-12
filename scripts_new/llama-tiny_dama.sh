python adapt_model.py --model_name "HuggingFaceM4/tiny-random-LlamaForCausalLM" --method "DAMA" --request_file "train_dama_tiny_processed.json" --num_layers 2 --post_linear True --iterative_update True --random_seed 45

# python evaluate_model.py --model_name "HuggingFaceM4/tiny-random-LlamaForCausalLM" --num_layers 2 --method "DAMA" --test_file test_dama.json --test_task gen
# python evaluate_model.py --model_name "HuggingFaceM4/tiny-random-LlamaForCausalLM" --num_layers 0 --method "MEMIT" --test_file "anti_type1_test.txt" --test_task coref
# python evaluate_model.py --model_name "HuggingFaceM4/tiny-random-LlamaForCausalLM" --num_layers 2 --method "MEMIT" --test_file stereoset_dev.json --test_task stereoset
# python evaluate_model.py --model_name "HuggingFaceM4/tiny-random-LlamaForCausalLM" --num_layers 2 --method "MEMIT" --test_file wikitext_wikitext-2-raw-v1 --test_task "causal_lm"
