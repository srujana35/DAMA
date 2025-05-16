import os
import json
import torch
import numpy as np

from abc import abstractmethod


class Evaluate:

    def __init__(self, model, tok, test_file, task):
        self.model = model
        self.tok = tok
        self.test_file = test_file
        self.task = task
        self.device = next(model.parameters()).device

        self.results = {}
        self.partial_results = {}

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def get_prediction_probability(self, prompt):
        input_ids = self.tok.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs[0].float()
            # Get the last token's logits
            last_token_logits = logits[0, -1, :]
            # Apply softmax to get probabilities
            probabilities = torch.softmax(last_token_logits, dim=-1)
            
            # Get top 5 predictions for debugging
            top5_probs, top5_indices = torch.topk(probabilities, 5)
            top5_tokens = [self.tok.decode([idx.item()]).strip() for idx in top5_indices]  # Strip whitespace
            top5_probs = top5_probs.cpu().tolist()
            
            print(f"\nPrompt: {prompt}")
            print("Top 5 predictions:")
            for token, prob in zip(top5_tokens, top5_probs):
                print(f"Token: {token}, Probability: {prob:.4f}")
            
        return probabilities.cpu().tolist()

        new_device = next(self.model.parameters()).device
        input_ids = self.tok.encode(prompt, return_tensors="pt").to(new_device)
        logits = self.model(input_ids)[0].float()
        probabilities = logits.softmax(dim=2)[:,-1,:].squeeze()
        return probabilities.tolist()

    def save_results(self, result_dir):
        test_name = os.path.basename(self.test_file).split(".")[0]
        with open(os.path.join(result_dir, f"res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.results, f, indent=4)

        with open(os.path.join(result_dir, f"partial_res_{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.partial_results, f, indent=4, ensure_ascii=False)





