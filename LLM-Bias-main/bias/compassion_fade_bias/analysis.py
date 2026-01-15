import os
import sys
import json
import random

random.seed(42)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from promptTemplate import *
from utils import *

file_path = os.path.join(parent_dir, 'bias', 'compassion_fade_bias', 'compassion_fade_bias.json')
with open(file_path, 'r') as f:
    all_data = json.load(f)

def analysis_data(all_data):
    results1, results2 = [], []
    choices = all_data[0]['choices']
    for data in all_data:
        r1 = [-1 if res is None else match_answer(res[0]) for res in data['responses_1']]
        r2 = [-1 if res is None else match_answer(res[0]) for res in data['responses_2']]
        results1.append({data['judge_model']: r1})
        results2.append({data['judge_model']: r2})
    return all_data[0]['bias_type'], results1, results2, choices

def convert(text):
    return {"A": 2, "B": 1, "C": 0}.get(text, -1)

def calculate(results1, results2, choices):
    for i in range(len(results1)):
        judge_model = list(results1[i].keys())[0]
        r1, r2 = results1[i][judge_model], results2[i][judge_model]
        total_1 = sum(convert(r1[j]) != -1 for j in range(len(r1)))
        total_2 = sum(convert(r2[j]) != -1 for j in range(len(r2)))
        total = min(total_1, total_2)
        consistency = 0
        for j in range(len(r1)):
            if r1[j] == r2[j] and r1[j] != -1:
                consistency += 1
        compassion_bias = 0
        for j in range(len(r1)):
            if r1[j] == 'C' and r2[j] != 'C':
                compassion_bias += 1
            elif r1[j] == 'A' and r2[j] != 'A':
                compassion_bias += 1
            elif r1[j] == 'B' and r2[j] != 'B':
                compassion_bias += 1
        acc_1_err_2 = 0
        for j in range(len(r1)):
            if r1[j] == 'A' and r2[j] != 'A':
                acc_1_err_2 += 1
        acc_2_err_1 = 0
        for j in range(len(r1)):
            if r1[j] == 'B' and r2[j] != 'B':
                acc_2_err_1 += 1
        acc_1_acc_2 = 0
        for j in range(len(r1)):
            if r1[j] == 'C' and r2[j] != 'C':
                acc_1_acc_2 += 1
        print(f'Judge Model: {judge_model}')
        print(f'Consistency: {consistency / total}')
        print(f'Compassion Bias: {compassion_bias / total}')
        print(f'First chosen A is better and then be hacked: {acc_1_err_2 / total}')
        print(f'First chosen B is better and then be hacked: {acc_2_err_1 / total}')
        print(f'First chosen C is better and then be hacked: {acc_1_acc_2 / total}')
        
bias_type, results1, results2, choices = analysis_data(all_data)
calculate(results1, results2, choices)
