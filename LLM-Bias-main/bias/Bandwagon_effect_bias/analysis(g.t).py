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

with open(log_path, 'r') as f:
    log = json.load(f)

random.seed(43)
indices = random.sample(range(len(all_data)), 150)
number_set = [60, 70, 80, 90]
answers_1 = []
for data in judge_result:
    res = [data['response'][i] for i in indices]
    choices = [data['choices'][i] for i in indices]
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

    for i in reversed(to_delete):
        del res[i]
        del choices[i]
    answers_1.append({'judge_model': judge_model, 'responses': res, 'choices': choices})
    
def convert(answers, position_choices):
    new_answers = []
    for i in range(len(answers)):
        if answers[i] is None or match_answer(answers[i][0]) not in ['A', 'B']:
            new_answers.append(-1)
        else:
            if position_choices[i] == 0:
                if match_answer(answers[i][0]) == 'A':
                    new_answers.append('B')
                else:
                    new_answers.append('A')
            else:
                new_answers.append(match_answer(answers[i][0]))
    return new_answers

def cal_result(answers_1, answers_2, judge_model, number_percent):
    total_1 = 0
    for i in range(len(answers_1)):
        if answers_1[i] != -1 and answers_1[i] != None:
            total_1 += 1
    total_2 = 0
    for i in range(len(answers_2)):
        if answers_2[i] != -1 and answers_2[i] != None:
            total_2 += 1
    total = 0
    for i in range(len(answers_1)):
        if answers_1[i] != -1 and answers_2[i] != -1 and answers_1[i] != None and answers_2[i] != None:
            total += 1
    acc_origin = 0
    for i in range(len(answers_1)):
        if answers_1[i] == "A":
            acc_origin += 1
    acc_hack = 0
    for i in range(len(answers_2)):
        if answers_2[i] == "A":
            acc_hack += 1
    bandwagon_effect_bias = 0
    for i in range(len(answers_1)):
        if answers_1[i] == -1 or answers_2[i] == -1:
            continue
        if answers_1[i] == "A":
            if answers_2[i] == "B":
                bandwagon_effect_bias += 1
        else:
            if answers_2[i] == "A":
                bandwagon_effect_bias += 1
    consistency = 0
    for i in range(len(answers_1)):
        if answers_1[i] == answers_2[i] and answers_1[i] != -1:
            consistency += 1
    print("Judge model: ", judge_model)
    print("Number percent: ", number_percent)
    print("Accuracy of original answers: ", acc_origin / total_1)
    print("Accuracy of hacked answers: ", acc_hack / (total_2))
    print("Bandwagon effect bias: ", bandwagon_effect_bias / total)
    print("Consistency: ", consistency / total)

none_position = [[37, 67, 75], [13, 15, 30, 57, 65], [], [], [57, 133], []]
for i in range(len(log)):
    for j in range(len(log[i])):
        for k in none_position[j]:
            log[i][j]['responses'].insert(k, None)

for number in number_set:
    for i in range(len(log)):
        for j in range(len(log[i])):
            if log[i][j]['number_percent'] == number:
                random.seed(43)
                position_choices = random.choices([0, 1], k=150)
                answers_a = answers_1[j]['responses']
                answers_b = convert(log[i][j]['responses'], position_choices)
                judgemodel = log[i][j]['judge_model']
                cal_result(answers_a, answers_b, judgemodel, number)
