import json
import torch
import os
from math import log, prod
from datasets import load_dataset

from evaluation import Evaluate

class EvaluateQA(Evaluate):
    def __init__(self, model, tok, test_file, task):
        super().__init__(model, tok, test_file, task)
        assert task == "qa", f"Task class mismatch:, expected 'qa', got '{task}' instead"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.load_data()

    def load_data(self):
        try:
            # Load OpenBookQA dataset from Hugging Face
            ds = load_dataset("allenai/openbookqa", "main")
            # Use only the test split
            self.dataset = ds['test']
            print(f"Loaded {len(self.dataset)} test examples from OpenBookQA")
            
            # Debug: Print the structure of the first example
            if len(self.dataset) > 0:
                print("\nDebug: First example structure:")
                print(json.dumps(self.dataset[0], indent=2))
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def save_results(self, output_dir):
        """Save evaluation results to a JSON file."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Use a fixed filename
            output_file = os.path.join(output_dir, "res_qa.json")
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            print(f"Results saved to {output_file}")
            
            # Also save partial results if they exist
            if hasattr(self, 'partial_results') and self.partial_results:
                partial_output_file = os.path.join(output_dir, "partial_res_qa.json")
                with open(partial_output_file, 'w') as f:
                    json.dump(self.partial_results, f, indent=2)
                print(f"Partial results saved to {partial_output_file}")
                
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            raise

    def evaluate(self):
        def score(a, n, norm):
            logx = lambda x: log(x) if x > 0 else float('-inf')
            return prod(a) / len(a), prod(a) / norm, sum([logx(x) for x in a]) - sum([logx(x) for x in n]), prod(a),  prod(a)**(1/len(a))

        partial_results = []
        for idx, d in enumerate(self.dataset):
            try:
                print(f"\nProcessing example {idx + 1}/{len(self.dataset)}")
                
                # Use the correct field names from the dataset
                prompt = d['question_stem']
                norm = "Answer:"
                prompt_tokens = self.tok.encode(prompt, return_tensors="pt").to(self.device)
                norm_tokens = self.tok.encode(norm, return_tensors="pt").to(self.device)
                ans = []
                correct = d['answerKey']
                
                # Debug: Print choices structure
                print(f"Choices structure: {d['choices']}")
                
                # Process choices where text and labels are in separate lists
                for i in range(len(d['choices']['text'])):
                    choice_text = d['choices']['text'][i]
                    choice_label = d['choices']['label'][i]
                    
                    choice_tokens = self.tok.encode(choice_text, return_tensors="pt").to(self.device)
                    choice_len = choice_tokens.shape[1]

                    # Ensure model output is on the same device
                    with torch.no_grad():
                        probs_q = torch.softmax(self.model.forward(torch.cat((prompt_tokens, choice_tokens), 1)).logits[:,-choice_len-1:-1,:].float(), dim=-1)[0, list(range(choice_len)), choice_tokens[0]]
                        probs_n = torch.softmax(self.model.forward(torch.cat((norm_tokens, choice_tokens), 1)).logits[:,-choice_len-1:-1,:].float(), dim=-1)[0, list(range(choice_len)), choice_tokens[0]]

                    # Move probabilities to CPU for scoring
                    probs_q = probs_q.cpu()
                    probs_n = probs_n.cpu()

                    answer_scores = score(probs_q.tolist(), probs_n.tolist(), len(choice_text))
                    ans.append((choice_label, answer_scores))
                
                partial_results.append((
                    sorted(ans, key=lambda x: x[1][0])[-1][0] == correct, 
                    sorted(ans, key=lambda x: x[1][1])[-1][0] == correct,
                    sorted(ans, key=lambda x: x[1][2])[-1][0] == correct,
                    sorted(ans, key=lambda x: x[1][3])[-1][0] == correct,
                    sorted(ans, key=lambda x: x[1][4])[-1][0] == correct
                ))
            except Exception as e:
                print(f"Error processing example {idx}: {str(e)}")
                print(f"Example data: {d}")
                continue

        if not partial_results:
            print("Warning: No results were generated")
            self.results = {
                'per_token_prob_root': 0,
                'per_token_prob': 0,
                'per_char_prob': 0,
                'normed_prob': 0,
                'unnormed_prob': 0,
            }
        else:
            self.results = {
                'per_token_prob_root': sum([x[4] for x in partial_results])/len(partial_results),
                'per_token_prob': sum([x[0] for x in partial_results])/len(partial_results),
                'per_char_prob': sum([x[1] for x in partial_results])/len(partial_results),
                'normed_prob': sum([x[2] for x in partial_results])/len(partial_results),
                'unnormed_prob': sum([x[3] for x in partial_results])/len(partial_results),
            }

        self.partial_results = {}
        