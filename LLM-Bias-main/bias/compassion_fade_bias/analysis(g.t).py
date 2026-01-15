import os
import sys
import json
import random

random.seed(42)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from promptTemplate import *
from utils import *

file_path = os.path.join(parent_dir, 'bias', 'compassion_fade_bias', 'compassion_fade_bias(g.t).json')
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
    return {"A": 1, "B": 0}.get(text, -1)

def calculate(results1, results2, choices):
    for i in range(len(results1)):
        judge_mdoel = list(results1[i].keys())[0]
        r1, r2 = results1[i][judge_mdoel], results2[i][judge_mdoel]
        total_1 = 0
        total_2 = 0
        acc_1 = 0
        for j in range(len(r1)):
            if convert(r1[j]) != -1:
                if convert(r1[j]) == choices[j]:
                    acc_1 += 1
                total_1 += 1
        acc_2 = 0
        for j in range(len(r2)):
            if convert(r2[j]) != -1:
                if convert(r2[j]) == choices[j]:
                    acc_2 += 1
                total_2 += 1
        total = min(total_1, total_2)
        acc_1_err_2 = 0
        for j in range(len(r1)):
            if convert(r1[j]) == choices[j] and convert(r2[j]) != choices[j] and convert(r2[j]) != -1:
                acc_1_err_2 += 1
        acc_2_err_1 = 0
        for j in range(len(r1)):
            if convert(r1[j]) != choices[j] and convert(r2[j]) == choices[j] and convert(r1[j]) != -1:
                acc_2_err_1 += 1
        compassion_fade_bias = 0
        for j in range(len(r1)):
            if r1[j] == 'A' and r2[j] != 'A':
                compassion_fade_bias += 1
            elif r1[j] == 'B' and r2[j] != 'B':
                compassion_fade_bias += 1

        print(f'Judge Model: {judge_mdoel}')
        print(f'Accuracy: {acc_1 / total_1}')
        print(f'Accuracy with compassion fade: {acc_2 / total_2}')
        print(f'Accuracy_1_error_2: {acc_1_err_2 / total}')
        print(f'Accuracy_2_error_1: {acc_2_err_1 / total}')
        print(f'Compassion fade bias: {compassion_fade_bias / total}') 

bias_type, results1, results2, choices = analysis_data(all_data)
print(len(results1), len(results2), len(choices))
calculate(results1, results2, choices)