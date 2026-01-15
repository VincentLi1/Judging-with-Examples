import os
import sys
import json
import random

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from promptTemplate import *
from utils import *

dataset_path = parent_dir + '/data/extensiveDataset.json'
log_path = parent_dir + '/bias/compassion_fade_bias/compassion_fade_bias.json'
new_path = parent_dir + '/bias/compassion_fade_bias/new.json'
with open(dataset_path, 'r') as f:
    all_data = json.load(f)

random.seed(43)
random_generate_model = random.sample(generate_model_set, 2) # ['mixtral-8x22b', 'llama3-70b']
questions = [data['question'] for data in all_data]
answers_1 = [response['response'][0]
             for data in all_data 
             for response in data['model_responses'] 
             if response['model'] == random_generate_model[0]]
answers_2 = [response['response'][0]
                for data in all_data
                for response in data['model_responses']
                if response['model'] == random_generate_model[1]]
models = list(model_dict.keys())
random.seed(42)
random_models_1 = random.choices(models, k=len(questions))
random.seed(43)
random_models_2 = random.choices(models, k=len(questions))
bias_type = 'compassion_fade_bias'
choices = random.choices([0, 1], k=len(questions))

def generate_pair_prompt(questions, answers_1, answers_2, random_models_1, random_models_2, choices):
    prompts_1, prompts_2 = [], []
    for i in range(len(questions)):
        if choices[i] == 1:
            prompts_1.append(compassion_fade_prompt(question=questions[i], answer_a=answers_1[i], answer_b=answers_2[i], need_tie=True))
            prompts_2.append(compassion_fade_prompt(question=questions[i], answer_a=answers_1[i], answer_b=answers_2[i], model_a=random_models_1[i], model_b=random_models_2[i], need_tie=True))
        else:
            prompts_1.append(compassion_fade_prompt(question=questions[i], answer_a=answers_2[i], answer_b=answers_1[i], need_tie=True))
            prompts_2.append(compassion_fade_prompt(question=questions[i], answer_a=answers_2[i], answer_b=answers_1[i], model_a=random_models_1[i], model_b=random_models_2[i], need_tie=True))
    return prompts_1, prompts_2

def generate_pair_judges(prompts_1, prompts_2, judge_model):
    prompts_1 = [(judge_model, prompt) for prompt in prompts_1]
    prompts_2 = [(judge_model, prompt) for prompt in prompts_2]
    responses_1 = model_function[judge_model](prompts_1)
    responses_2 = model_function[judge_model](prompts_2)
    responses_1 = [response for response in responses_1]
    responses_2 = [response for response in responses_2]
    return responses_1, responses_2

prompts_1, prompts_2 = generate_pair_prompt(questions, answers_1, answers_2, random_models_1, random_models_2, choices)

# judge_result = []
with open(log_path, 'r') as f:
    judge_result = json.load(f)
    
for judge_model in judge_model_set:
    responses_1, responses_2 = generate_pair_judges(prompts_1, prompts_2, judge_model)
    judge_result.append({
        'judge_model': judge_model,
        'responses_1': responses_1,
        'responses_2': responses_2,
        'choices': choices,
        'bias_type': bias_type
    })
    print(f'Finished {judge_model}')

with open(log_path, 'w') as f:
    json.dump(judge_result, f, indent=4)