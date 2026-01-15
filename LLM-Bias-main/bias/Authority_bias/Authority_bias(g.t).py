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
references_path = parent_dir + '/bias/Authority_bias/references.json'
log_path = parent_dir + '/bias/Authority_bias/Authority_bias(g.t).json'
with open(dataset_path, 'r') as f:
    all_data = json.load(f)

with open(judge_reslut_path, 'r') as f:
    judge_result = json.load(f)

with open(references_path, 'r') as f:
    references = json.load(f)

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

    answers_1.append({'judge_model': judge_model, 'responses': res, 'choices': choices})

questions = [data['question'] for data in sample_all_data]
chosen = [data['chosen'] for data in sample_all_data]
rejected = [data['rejected'] for data in sample_all_data]
random.seed(43)
position_choices = random.choices([0, 1], k=len(questions))
bias_type = 'Authority_bias'
reference_types = ['book', 'quote', 'url']
def generate_prompt(questions, chosen, rejected, chosen_reference, rejected_reference, answers):
    prompts = []
    for i in range(len(questions)):
        if answers[i] is None or answers[i] not in ['A', 'B']:
            continue
        if position_choices[i] == 1:
            if answers[i][0] == 'A':
                prompts.append(Authority_bias_prompt(question=questions[i], answer_a=chosen[i], answer_b=rejected[i], reference=rejected_reference[i], hack_answer='B'))
            else:
                prompts.append(Authority_bias_prompt(question=questions[i], answer_a=chosen[i], answer_b=rejected[i], reference=chosen_reference[i], hack_answer='A'))
        else:
            if answers[i][0] == 'A':
                prompts.append(Authority_bias_prompt(question=questions[i], answer_a=rejected[i], answer_b=chosen[i], reference=rejected_reference[i], hack_answer='A'))
            else:
                prompts.append(Authority_bias_prompt(question=questions[i], answer_a=rejected[i], answer_b=chosen[i], reference=chosen_reference[i], hack_answer='B'))
    return prompts

results = []
for type in reference_types:
    chosen_reference = [data[f'{type}_reference_chosen'] for data in references]
    rejected_reference = [data[f'{type}_reference_rejected'] for data in references]
    type_result = []
    for judge_model in judge_model_set:
        ans_1 = [data['responses'] for data in answers_1 if data['judge_model'] == judge_model][0]
        prompts = generate_prompt(questions, chosen, rejected, chosen_reference, rejected_reference, ans_1)
        mix_prompts = [(judge_model, prompt) for prompt in prompts]
        responses = model_function[judge_model](mix_prompts)
        type_result.append(
            {
                'judge_model': judge_model,
                'responses': responses,
                'reference_type': type
            }
        )
    results.append(type_result)

with open(log_path, 'w') as f:
    json.dump(results, f, indent=4)