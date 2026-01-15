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
log_path = parent_dir + '/bias/random_test/random_dataset_1.json'
with open(dataset_path, 'r') as f:
    all_data = json.load(f)

with open(judge_reslut_path, 'r') as f:
    judge_result = json.load(f)

questions = [data['question'] for data in all_data]
chosen = [data['chosen'] for data in all_data]
rejected = [data['rejected'] for data in all_data]
position_choices = judge_result[0]['choices']

def convert(answers):
    new_answers = []
    for i in range(len(answers)):
        if answers[i] is None or match_answer(answers[i][0]) not in ['A', 'B']:
            new_answers.append(None)
        else:
            if position_choices[i] == 1:
                if match_answer(answers[i][0]) == 'A':
                    new_answers.append('B')
                else:
                    new_answers.append('A')
            else:
                new_answers.append(match_answer(answers[i][0]))
    return new_answers

answers_1 = []
for data in judge_result:
    res = convert(data['response'])
    answers_1.append(res)

rs = []
results = []
for idx, model in enumerate(judge_model_set):
    prompts = []
    for i in range(len(questions)):
        if position_choices[i] == 1:
            prompts.append(evaluate_ai_responses_no_tie(questions[i], chosen[i], rejected[i]))
        else:
            prompts.append(evaluate_ai_responses_no_tie(questions[i], rejected[i], chosen[i]))
    q = [(model, prompt) for prompt in prompts]
    responses = model_function[model](q)
    res = convert(responses)
    total = 0
    for i in range(len(res)):
        if res is not None and answers_1[idx][i] is not None:
            total += 1
    consistency = 0
    for i in range(len(res)):
        if res is not None and answers_1[idx][i] is not None and res[i] == answers_1[idx][i]:
            consistency += 1
    print(model, consistency / total)
    rs.append({'judge_model': model, 'consistency': consistency / total})
    results.append({'judge_model': model, 'responses': res})

with open(log_path, 'w') as f:
    json.dump(results, f, indent=4)

print(rs)