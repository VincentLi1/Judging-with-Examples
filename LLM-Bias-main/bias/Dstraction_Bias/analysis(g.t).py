import os
import sys
import json
import random

random.seed(42)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from promptTemplate import *
from utils import *
judge_reslut_path = parent_dir + '/data/judge_result.json'

file_path = os.path.join(parent_dir, 'bias', 'Dstraction_Bias', 'Modified_Dstraction_Bias(g.t)_B.json')
with open(file_path, 'r', encoding='utf-8') as f:
    all_data = json.load(f)
with open(judge_reslut_path, 'r', encoding='utf-8') as f:
    judge_result = json.load(f)

print(len(all_data[4]['responses']))
print(len(judge_result[0]['response']))
def analysis_data(all_data):
    results1=[]
    choices = all_data[0]['choices']
    for data in all_data:
        r1 = [-1 if res is None else match_answer(res[0]) for res in data['response']]
        results1.append({data['judge_model']: r1})
            #print(r1)
        # print(r1)
    return results1, choices

answers_1 = []
for data in judge_result:
    res = data['response']
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
    answers_1.append({'judge_model': judge_model, 'responses': res, 'choices': choices})

answers_2 = []
for data in all_data:
    res = data['responses']
    choices = data['choices']
    judge_model = data['judge_model']
    to_delete = []
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
    answers_2.append({'judge_model': judge_model, 'responses': res, 'choices': choices})


for i in range(len(answers_1)):
    judge_model = answers_1[i]['judge_model']
    total_1 = 0
    total_2 = 0
    for j in range(len(answers_1[i]['responses'])):
        if answers_2[i]['responses'][j] is not None and answers_1[i]['responses'][j] is not None and answers_1[i]['choices'][j] == 0:
            total_1 += 1
        if answers_2[i]['responses'][j] is not None and answers_1[i]['responses'][j] is not None and answers_1[i]['choices'][j] == 1:
            total_2 += 1
    consistency_a = 0
    accuracy_a_origin = 0
    accuracy_a_hack = 0
    for j in range(len(answers_1[i]['responses'])):
        if answers_1[i]['responses'][j] is not None and answers_2[i]['responses'][j] is not None and answers_1[i]['choices'][j] == 0:
            if answers_1[i]['responses'][j][0] == 'A':
                accuracy_a_origin += 1
    for j in range(len(answers_1[i]['responses'])):
        if answers_1[i]['responses'][j] is not None and answers_2[i]['responses'][j] is not None and answers_1[i]['choices'][j] == 0:
            if answers_2[i]['responses'][j][0] == 'A':
                accuracy_a_hack += 1
    for j in range(len(answers_1[i]['responses'])):
        if answers_1[i]['responses'][j] is not None and answers_2[i]['responses'][j] is not None and answers_1[i]['choices'][j] == 0:
            if answers_1[i]['responses'][j][0] == answers_2[i]['responses'][j][0]:
                consistency_a += 1
    consistency_b = 0
    accuracy_b_origin = 0
    accuracy_b_hack = 0
    for j in range(len(answers_1[i]['responses'])):
        if answers_1[i]['responses'][j] is not None and answers_2[i]['responses'][j] is not None and answers_1[i]['choices'][j] == 1:
            if answers_1[i]['responses'][j][0] == 'A':
                accuracy_b_origin += 1
    for j in range(len(answers_1[i]['responses'])):
        if answers_1[i]['responses'][j] is not None and answers_2[i]['responses'][j] is not None and answers_1[i]['choices'][j] == 1:
            if answers_2[i]['responses'][j][0] == 'A':
                accuracy_b_hack += 1
    for j in range(len(answers_1[i]['responses'])):
        if answers_1[i]['responses'][j] is not None and answers_2[i]['responses'][j] is not None and answers_1[i]['choices'][j] == 1:
            if answers_1[i]['responses'][j][0] == answers_2[i]['responses'][j][0]:
                consistency_b += 1
    print('judge_model:', judge_model)
    # print('total_1:', total_1)
    # print('total_2:', total_2)
    # print('accuracy_a_origin:', accuracy_a_origin / total_1)
    # print('accuracy_a_hack:', accuracy_a_hack / total_1)
    print('consistency_a:', consistency_a / total_1)
    # print('accuracy_b_origin:', accuracy_b_origin / total_2)
    # print('accuracy_b_hack:', accuracy_b_hack / total_2)
    print('consistency_b:', consistency_b / total_2)
