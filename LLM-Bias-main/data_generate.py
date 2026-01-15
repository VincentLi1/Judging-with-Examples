import json
import pandas as pd
from utils import *
from promptTemplate import *

data_path = "./data/extensiveDataset.json"
output_path = './data/AllDataset.json'

try:
    with open(data_path, 'r') as f:
        all_data = json.load(f)
except:
    all_data = []   
questions = [data['question'] for data in all_data]
model_responses = [{} for _ in range(len(questions))]

for key in model_dict:
    prompts = [(key, question) for question in questions]
    responses = model_function[key]
    for i in range(len(responses)):
        model_responses[i][key] = responses[i]
    print(f'Finished {key}')

final_data = []
for i in range(len(questions)):
    data = {
        'question': questions[i],
        'chosen': all_data[i]['chosen'],
        'rejected': all_data[i]['rejected'],
        'data_resource': all_data[i]['dataset_resource'],
        'model_responses': [{'model': key, 'response': model_responses[i][key]} for key in model_dict]
    }
    final_data.append(data)

with open(output_path, 'w') as f:
    json.dump(final_data, f, indent=4)