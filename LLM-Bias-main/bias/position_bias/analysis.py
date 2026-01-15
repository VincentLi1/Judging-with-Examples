import os
import sys
import json
import random

random.seed(42)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from promptTemplate import *
from utils import *

file_path = os.path.join(parent_dir, 'bias', 'position_bias', 'position_bias(g.t).json')
with open(file_path, 'r') as f:
    test_data = json.load(f)

def analysis_data(test_data):
    results1, results2 = [], []
    choices = test_data[0]['choises']
    for data in test_data:
        r1 = [-1 if res is None else match_answer(res[0]) for res in data['responses_1']]
        r2 = [-1 if res is None else match_answer(res[0]) for res in data['responses_2']]
        results1.append({data['judge_model']: r1})
        results2.append({data['judge_model']: r2})
    return test_data[0]['bias_type'], test_data[0]['judge_model'], results1, results2, choices

def convert(text):
    return {"A": 1, "B": 0}.get(text, -1)

def calculate(results1, results2, choices):
    for i in range(len(results1)):
        judge_model = list(results1[i].keys())[0]
        r1, r2 = results1[i][judge_model], results2[i][judge_model]
        acc = sum(convert(r1[j]) == choices[j] for j in range(len(r1)) if convert(r1[j]) != -1)
        acc_r = sum(convert(r2[j]) == 1 - choices[j] for j in range(len(r2)) if convert(r2[j]) != -1)
        total = sum(convert(r1[j]) != -1 for j in range(len(r1)))
        total_r = sum(convert(r2[j]) != -1 for j in range(len(r2)))
        pos_bias = sum(r1[j] == r2[j] and r1[j] != -1 for j in range(len(r1)))
        acc_1_err_2 = sum(convert(r1[j]) == choices[j] and convert(r2[j]) != 1 - choices[j] and convert(r2[j]) != -1 for j in range(len(r1)))
        acc_2_err_1 = sum(convert(r1[j]) != choices[j] and convert(r2[j]) == 1 - choices[j] and convert(r1[j]) != -1 for j in range(len(r1)))
        
        print(f'Judge Model: {judge_model}')
        print(f'Accuracy: {acc / total}')
        print(f'Accuracy_r: {acc_r / total_r}')
        print(f'Position Bias: {pos_bias / min(total, total_r)}')
        print(f'Accuracy_1_error_2: {acc_1_err_2 / min(total, total_r)}')
        print(f'Accuracy_2_error_1: {acc_2_err_1 / min(total, total_r)}')

bias_type, judge_model, results1, results2, choices = analysis_data(test_data)
calculate(results1, results2, choices)
