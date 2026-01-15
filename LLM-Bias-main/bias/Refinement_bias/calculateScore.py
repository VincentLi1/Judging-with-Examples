import json
import re
from collections import defaultdict

def extract_score(value):
    if isinstance(value, list) and len(value) > 0:
        if isinstance(value[-1], str):
            match = re.search(r'Rating:\s*\[*\[*(\d+)\]*\]*', value[-1])
            if match:
                return int(match.group(1))
        elif len(value) > 1 and isinstance(value[-2], str):
            match = re.search(r'Rating:\s*\[*\[*(\d+)\]*\]*', value[-2])
            if match:
                return int(match.group(1))
    elif isinstance(value, str):
        match = re.search(r'Rating:\s*\[*\[*(\d+)\]*\]*', value)
        if match:
            return int(match.group(1))
    return None

# 读取JSON文件
with open('answers_with_polish_and_scores.json', 'r') as f:
    data = json.load(f)

# 初始化字典来存储每个模型的不同类型的评分
scores = defaultdict(lambda: defaultdict(list))

# 遍历JSON数据
for item in data:
    for key, value in item.items():
        if key.endswith('_score'):
            model_name = key.split('_')[0]
            score_type = '_'.join(key.split('_')[1:-1])  # 获取 'initial', 'polished', 或 'full_conversation'
            score = extract_score(value)
            if score is not None:
                scores[model_name][score_type].append(score)

# 计算平均分
average_scores = {}
for model, score_types in scores.items():
    average_scores[model] = {}
    for score_type, score_list in score_types.items():
        if score_list:
            average_scores[model][score_type] = sum(score_list) / len(score_list)

# 打印结果
for model, score_types in average_scores.items():
    print(f"\n{model}:")
    for score_type, avg_score in score_types.items():
        print(f"  {score_type}: {avg_score:.2f}")

# 将平均分保存为JSON文件
with open('average_scores.json', 'w') as f:
    json.dump(average_scores, f, indent=2)

print("\n平均分已保存到 'average_scores.json' 文件中。")