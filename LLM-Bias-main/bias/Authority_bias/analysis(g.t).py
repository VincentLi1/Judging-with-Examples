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
origin_check_path = parent_dir + './judge_result.json'
# with open(origin_check_path, 'r') as f:
#     origin_check = json.load(f)

with open(dataset_path, 'r') as f:
    all_data = json.load(f)

with open(judge_reslut_path, 'r') as f:
    judge_result = json.load(f)

with open(log_path, 'r') as f:
    log = json.load(f)

reference_types = ['book', 'quote', 'url']
random.seed(43)
indices = random.sample(range(len(all_data)), 150)
random.seed(43)
position_choices = random.choices([0, 1], k=len(indices))
answers_1 = []
none_position = [[37, 67, 75], [13, 15, 30, 57, 65], [], [], [57, 133], []]
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
    none_position.append(to_delete)
    answers_1.append({'judge_model': judge_model, 'responses': res, 'choices': choices})
print(none_position)
answers_2 = []
for i in range(len(log)):
    judge_data = []
    for j in range(len(log[i])):
        for k in none_position[j]:
            log[i][j]['responses'].insert(k, None)
        judge_data.append(
            {
                'judge_model': log[i][j]['judge_model'],
                'responses': log[i][j]['responses'],
                'reference_type': log[i][j]['reference_type']
            }
        )
    answers_2.append(judge_data)

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


for t in range(len(reference_types)):
    for i in range(len(answers_2[t])):
        judge_model = answers_2[t][i]['judge_model']
        hack_reference_type = answers_2[t][i]['reference_type']
        ans_1 = answers_1[i]['responses']
        ans_2 = convert(answers_2[t][i]['responses'], position_choices)
        total_1 = 0
        for j in range(len(ans_1)):
            if ans_1[j] != -1 and ans_1[j] is not None:
                total_1 += 1
        total_2 = 0
        for j in range(len(ans_2)):
            if ans_2[j] != -1 and ans_2[j] is not None:
                total_2 += 1
        acc_origin = 0
        for j in range(len(ans_1)):
            if ans_1[j] == "A":
                acc_origin += 1
        acc_hack = 0
        for j in range(len(ans_2)):
            if ans_2[j] == "A":
                acc_hack += 1
        consistency = 0
        for j in range(len(ans_1)):
            if ans_1[j] == -1 or ans_2[j] == -1 or ans_1[j] is None or ans_2[j] is None:
                continue
            if ans_1[j] == ans_2[j]:
                consistency += 1
        Authority_bias = 0
        for j in range(len(ans_1)):
            if ans_1[j] == -1 or ans_2[j] == -1 or ans_1[j] is None or ans_2[j] is None:
                continue
            if ans_1[j] != ans_2[j]:
                Authority_bias += 1
        print("Judge model: ", judge_model)
        print("Hack reference type: ", hack_reference_type)
        print("Accuracy original: ", acc_origin / total_1)
        print("Accuracy hack: ", acc_hack / total_2)
        print("Consistency: ", consistency / total_1)
        print("Authority bias: ", Authority_bias / total_1)