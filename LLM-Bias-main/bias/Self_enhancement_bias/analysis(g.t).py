import os
import sys
import json
import re
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from promptTemplate import *
from utils import *

dataset_path = parent_dir + '/data/extensiveDataset.json'
result_path = parent_dir + '/bias/Self_enhancement_bias/Self_enhancement_bias(g.t).json'

with open(dataset_path, 'r') as f:
    all_data = json.load(f)

with open(result_path, 'r') as f:
    judge_result = json.load(f)

def extract_rating(text):
    pattern = re.compile(r"Rating: \[\[(\d+)\]\]|\bRating: (\d+)\b")
    match = pattern.search(text)
    if match:
        return int(match.group(1) if match.group(1) else match.group(2))
    return None

all_ratings = []
for judge_model_data in judge_result:
    for test_model_data in judge_model_data['judge_results']:
        for response in test_model_data['responses']:
            rating = extract_rating(response[0]) if response is not None else None
            if rating is not None:
                all_ratings.append(rating)

mean_rating = np.mean(all_ratings)
std_rating = np.std(all_ratings)
min_rating = np.min(all_ratings)
max_rating = np.max(all_ratings)

def z_score_normalize(rating, mean_rating, std_rating):
    return (rating - mean_rating) / std_rating

def min_max_normalize(rating, min_rating, max_rating):
    return (rating - min_rating) / (max_rating - min_rating)

for judge_model_data in judge_result:
    judge_model = judge_model_data['judge_model']
    print('------------------------------------------')
    print("Judge model: ", judge_model)
    for test_model_data in judge_model_data['judge_results']:
        test_model = test_model_data['test_model']
        ratings = []
        for response in test_model_data['responses']:
            if response is None:
                ratings.append(None)
            else:
                ratings.append(extract_rating(response[0]))
        
        avg_rating = sum([r for r in ratings if r is not None]) / len([r for r in ratings if r is not None])
        normalized_avg_rating_zscore = z_score_normalize(avg_rating, mean_rating, std_rating)
        normalized_avg_rating_minmax = min_max_normalize(avg_rating, min_rating, max_rating)
        
        print("Test model: ", test_model)
        print("Average rating: ", avg_rating)
        print("Z-score Normalized average rating: ", normalized_avg_rating_zscore)
        print("Min-Max Normalized average rating: ", normalized_avg_rating_minmax)
