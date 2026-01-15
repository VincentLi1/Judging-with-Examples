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
log_path = parent_dir + '/bias/Authority_bias/references.json'
with open(dataset_path, 'r') as f:
    all_data = json.load(f)

with open(judge_reslut_path, 'r') as f:
    judge_result = json.load(f)

random.seed(43)
indices = random.sample(range(len(all_data)), 150)
sample_all_data = [all_data[i] for i in indices]

questions = [data['question'] for data in sample_all_data]
chosen = [data['chosen'] for data in sample_all_data]
rejected = [data['rejected'] for data in sample_all_data]

references = []
for i in range(len(chosen)):
    book_reference_chosen = generate_book_prompt(chosen[i])
    book_reference_rejected = generate_book_prompt(rejected[i])
    quote_reference_chosen = generate_quote_prompt(chosen[i])
    quote_reference_rejected = generate_quote_prompt(rejected[i])
    url_reference_chosen = generate_url_prompt(chosen[i])
    url_reference_rejected = generate_url_prompt(rejected[i])
    prompts = [('claude-3.5-sonnet', book_reference_chosen), ('claude-3.5-sonnet', book_reference_rejected), ('claude-3.5-sonnet', quote_reference_chosen), ('claude-3.5-sonnet', quote_reference_rejected), ('claude-3.5-sonnet', url_reference_chosen), ('claude-3.5-sonnet', url_reference_rejected)]
    responses = get_multiple_other_model_responses(prompts)
    book_reference_chosen = extract_bracket_content(responses[0][0])
    book_reference_rejected = extract_bracket_content(responses[1][0])
    quote_reference_chosen = extract_bracket_content(responses[2][0])
    quote_reference_rejected = extract_bracket_content(responses[3][0])
    url_reference_chosen = extract_bracket_content(responses[4][0])
    url_reference_rejected = extract_bracket_content(responses[5][0])
    references.append(
        {
            'question': questions[i],
            'chosen': chosen[i],
            'rejected': rejected[i],
            'book_reference_chosen': book_reference_chosen,
            'book_reference_rejected': book_reference_rejected,
            'quote_reference_chosen': quote_reference_chosen,
            'quote_reference_rejected': quote_reference_rejected,
            'url_reference_chosen': url_reference_chosen,
            'url_reference_rejected': url_reference_rejected
        }
    )

with open(log_path, 'w') as f:
    json.dump(references, f, indent=4)