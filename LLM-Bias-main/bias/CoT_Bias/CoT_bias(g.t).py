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
log_path = parent_dir + '/bias/CoT_Bias/CoT_Bias(g.t).json'
with open(dataset_path, 'r', encoding='utf-8') as f:
    all_data = json.load(f)

with open(judge_reslut_path, 'r', encoding='utf-8') as f:
    judge_result = json.load(f)

random.seed(43)
#indices = random.sample(range(len(all_data)), 150)
sample_all_data = all_data
answers_1 = []
non_position = []
for data in judge_result:
    res = data['responses']
    choices = data['choices']
    judge_model = data['judge_model']
    to_delete = []
    for i in range(len(res)):
        if res[i] is None or match_answer(res[i][0]) not in ['A', 'B']:
            res[i] = None
            to_delete.append(i)
        else:
            res[i] = match_answer(res[i][0])
            if choices[i] == 0:
                if res[i] == 'A':
                    res[i] = 'B'
                else:
                    res[i] = 'A'
    non_position.append(to_delete)
    answers_1.append({'judge_model': judge_model, 'response': res, 'choices': choices})
print(non_position)
questions = [data['question'] for data in sample_all_data]
chosen = [data['chosen'] for data in sample_all_data]
rejected = [data['rejected'] for data in sample_all_data]
models = list(model_dict.keys())
random.seed(43)
position_choices = random.choices([0, 1], k=len(questions))
bais_type = 'CoT_Bias'

def generate_prompt(questions, chosen, rejected, position_choices, number, answers):
    prompts = []
    choice_dict = {
        'A': "Assistant B",
        'B': "Assistant A"
    }
    for i in range(len(questions)):
        if answers[i] is None:
            # print(i)
            continue
        if position_choices[i] == 1:
            prompts.append(CotPrompt(question=questions[i], answer_a=chosen[i], answer_b=rejected[i]))
        else:
            prompts.append(CotPrompt(question=questions[i], answer_a=rejected[i], answer_b=chosen[i]))
    return prompts

def generate_judges(prompts, judge_model):
    prompts = [(judge_model, prompt) for prompt in prompts]
    responses = model_function[judge_model](prompts)
    responses = [response for response in responses]
    return responses

final_results = []
results = []
i=0
for index, judge_model in enumerate(judge_model_set):
    prompts = generate_prompt(questions, chosen, rejected, position_choices, 60, answers_1[index]['response'])
    print(len(prompts))
#     responses = generate_judges(prompts, judge_model)
#     results.append({
#         'judge_model': judge_model,
#         'responses': responses,
#         'id': index,
#         'choices': position_choices
#     })
#     print(f'Finished {judge_model}')
# final_results.append(results)

# with open(log_path, 'w') as f:
#     json.dump(final_results, f, indent=4)

