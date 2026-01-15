import os
import sys
import json
import random

random.seed(42)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from promptTemplate import *
from utils import *

dataset_path = parent_dir + '/data/extensiveDataset.json'
log_path = parent_dir + '/bias/position_bias/position_bias(g.t).json'

with open(dataset_path, 'r') as f:
    all_data = json.load(f)

questions = [data['question'] for data in all_data]
chosen = [data['chosen'] for data in all_data]
rejected = [data['rejected'] for data in all_data]

def generate_pair_prompt(chosen, rejected, questions):
    prompts_1 = []
    prompts_2 = []
    choises = []
    for i in range(len(questions)):
        choise = 1
        if random.random() > 0.5:
            choise = 0
        if choise == 1:
            prompts_1.append(evaluate_ai_responses(question=questions[i], answer_a=chosen[i], answer_b=rejected[i]))
            prompts_2.append(evaluate_ai_responses(question=questions[i], answer_a=rejected[i], answer_b=chosen[i]))
        else:
            prompts_1.append(evaluate_ai_responses(question=questions[i], answer_a=rejected[i], answer_b=chosen[i]))
            prompts_2.append(evaluate_ai_responses(question=questions[i], answer_a=chosen[i], answer_b=rejected[i]))
        choises.append(choise)
    return prompts_1, prompts_2, choises

def generate_pair_judges(prompts_1, prompts_2, judge_model):
    prompts_1 = [(judge_model, prompt) for prompt in prompts_1]
    prompts_2 = [(judge_model, prompt) for prompt in prompts_2]
    responses_1 = model_function[judge_model](prompts_1)
    responses_2 = model_function[judge_model](prompts_2)
    responses_1 = [response for response in responses_1]
    responses_2 = [response for response in responses_2]
    return responses_1, responses_2

judge_result = []
prompts_1, prompts_2, choises = generate_pair_prompt(chosen, rejected, questions)
for judge_model in judge_model_set:
    responses_1, responses_2 = generate_pair_judges(prompts_1, prompts_2, judge_model)
    judge_result.append({
        'judge_model': judge_model,
        'responses_1': responses_1,
        'responses_2': responses_2,
        'choises': choises,
        'bias_type': 'position_bias'
    })
    print(f'Finished {judge_model}')

with open(log_path, 'w') as f:
    json.dump(judge_result, f, indent=4)
