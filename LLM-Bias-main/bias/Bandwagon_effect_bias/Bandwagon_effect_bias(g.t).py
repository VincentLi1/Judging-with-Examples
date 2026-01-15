import os
import sys
import json
import random

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from promptTemplate import *
from utils import *

dataset_path = parent_dir + '/data/extensiveDataset.json'
judge_reslut_path = parent_dir + '/data/judge_result.json'
log_path = parent_dir + '/bias/Bandwagon_effect_bias/Bandwagon_effect_bias(g.t).json'
with open(dataset_path, 'r') as f:
    all_data = json.load(f)

with open(judge_reslut_path, 'r') as f:
    judge_result = json.load(f)

random.seed(43)
indices = random.sample(range(len(all_data)), 150)
sample_all_data = [all_data[i] for i in indices]
answers_1 = []
for data in judge_result:
    res = [data['response'][i] for i in indices]
    choices = [data['choices'][i] for i in indices]
    judge_model = data['judge_model']
    for i in range(len(res)):
        if res[i] is None or match_answer(res[i][0]) not in ['A', 'B']:
            res[i] = None
        else:
            res[i] = match_answer(res[i][0])
            if choices[i] == 0:
                if res[i] == 'A':
                    res[i] = 'B'
                else:
                    res[i] = 'A'
    answers_1.append({'judge_model': judge_model, 'response': res, 'choices': choices})

questions = [data['question'] for data in sample_all_data]
chosen = [data['chosen'] for data in sample_all_data]
rejected = [data['rejected'] for data in sample_all_data]
models = list(model_dict.keys())
random.seed(43)
position_choices = random.choices([0, 1], k=len(questions))
bais_type = 'Bandwagon_effect_bias'
number_set = [60, 70, 80, 90]

def generate_prompt(questions, chosen, rejected, position_choices, number, answers):
    prompts = []
    choice_dict = {
        'A': "Assistant B",
        'B': "Assistant A"
    }
    for i in range(len(questions)):
        if answers[i] is None:
            continue
        if position_choices[i] == 1:
            prompts.append(Bandwagon_effect_prompt(question=questions[i], answer_a=chosen[i], answer_b=rejected[i], number=number, chosen_model=choice_dict[answers[i]]))
        else:
            prompts.append(Bandwagon_effect_prompt(question=questions[i], answer_a=rejected[i], answer_b=chosen[i], number=number, chosen_model=f'Assistant {answers[i]}'))
    return prompts

def generate_judges(prompts, judge_model):
    prompts = [(judge_model, prompt) for prompt in prompts]
    responses = model_function[judge_model](prompts)
    responses = [response for response in responses]
    return responses

final_results = []
for number in number_set:
    results = []
    for index, judge_model in enumerate(judge_model_set):
        prompts = generate_prompt(questions, chosen, rejected, position_choices, 60, answers_1[index]['response'])
        responses = generate_judges(prompts, judge_model)
        results.append({
            'judge_model': judge_model,
            'responses': responses,
            'bias_type': bais_type,
            'number_percent': number,
            'choices': position_choices
        })
        print(f'Finished {judge_model}')
    final_results.append(results)

with open(log_path, 'w') as f:
    json.dump(final_results, f, indent=4)

