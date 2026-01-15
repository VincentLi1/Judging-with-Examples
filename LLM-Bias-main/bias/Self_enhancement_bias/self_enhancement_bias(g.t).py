import os
import sys
import json
import random

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from promptTemplate import *
from utils import *

dataset_path = parent_dir + '/data/extensiveDataset.json'
log_path = parent_dir + '/bias/self_enhancement_bias/self_enhancement_bias(g.t).json'
with open(dataset_path, 'r') as f:
    all_data = json.load(f)

random.seed(42)
indices = random.sample(range(len(all_data)), 100)
sample_all_data = [all_data[i] for i in indices]

questions = [data['question'] for data in sample_all_data]
ground_truths = [data['chosen'] for data in sample_all_data]
model_responses = [{"model" : model, "responses":[]} for model in judge_model_set]

# print(sample_all_data[0])
for data in sample_all_data:
    for res in data['model_responses']:
        for model in model_responses:
            if model['model'] == res['model']:
                model['responses'].append(res['response'][0])

def generate_prompt(questions, ground_truths, answer):
    prompts = []
    for i in range(len(questions)):
        prompt = self_enhancement_bias_prompt(question=questions[i], ground_truth=ground_truths[i], answer=answer[i])
        prompts.append(prompt)
    return prompts

results = []
for i in range(len(judge_model_set)):
    judge_model = judge_model_set[i]
    bias_type = 'Self_enhancement_bias'

    judge_results = []
    for j in range(len(judge_model_set)):
        test_model = judge_model_set[j]
        prompts = generate_prompt(questions, ground_truths, model_responses[j]['responses'])
        mix_prompts = [(judge_model, prompt) for prompt in prompts]
        responses = model_function[judge_model](mix_prompts)
        judge_results.append(
            {"test_model": test_model, "responses": responses}
        )
    results.append(
        {"judge_model": judge_model, "judge_results": judge_results}
    )

with open(log_path, 'w') as f:
    json.dump(results, f, indent=4)