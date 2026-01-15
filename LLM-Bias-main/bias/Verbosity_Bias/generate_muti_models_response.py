import json
from promptTemplate import evaluate_ai_responses_no_tie
from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

judge_models = ['gpt-35-turbo', 'gpt-4', 'gpt-4o', 'GLM-4', 'claude-3.5-sonnet', 'Qwen2-72b']

# 读取问题和答案
with open('resultsFormattedLonger.json', 'r') as file:
    data = json.load(file)

results = []
output_path = 'data/Reverse_evaluationResults.json'

# 尝试从现有文件加载已有的结果
try:
    with open(output_path, 'r') as file:
        results = json.load(file)
    start_id = len(results)
except FileNotFoundError:
    start_id = 0

def evaluate_model(key, evaluate_prompt):
    # 根据模型类型调用不同的API
    if key in ['gpt-35-turbo', 'gpt-4', 'gpt-4o']:
        return get_multiple_openai_responses([(key, evaluate_prompt)])
    elif key in ['GLM-4', 'claude-3.5-sonnet']:
        return get_multiple_other_model_responses([(key, evaluate_prompt)])
    elif key in ["mixtral-8x22b", "Qwen2-72b", "llama3-70b"]:
        return get_multiple_large_model_responses([(key, evaluate_prompt)])
    elif key in ["llama3-8b"]:
        return get_multiple_opensource_model_responses([(key, evaluate_prompt)])
    else:
        return None

# 创建线程池执行器
with ThreadPoolExecutor(max_workers=len(judge_models)) as executor:
    # 修改迭代步进为2，同时处理两个条目
    for idx in range(start_id, len(data), 4):
        tasks = []
        for sub_idx in [0, 1, 2, 3]:  # 处理当前和下一个条目和下下个
            if idx + sub_idx >= len(data):
                continue
            item = data[idx + sub_idx]
            if 'model_responses' not in item or len(item['model_responses']) < 2:
                continue

            question = item["question"]
            #answer_a = item["model_responses"][0]['response'][0]
            #answer_b = item["model_responses"][2]['response'][0][0]

            answer_a = item["model_responses"][1]['response'][0]
            answer_b = item["model_responses"][0]['response'][0]
            evaluate_prompt = evaluate_ai_responses_no_tie(question, answer_a, answer_b)

            future_to_key = {executor.submit(evaluate_model, key, evaluate_prompt): key for key in judge_models}
            tasks.append((future_to_key, idx + sub_idx, question, answer_a, answer_b))
        
        for future_to_key, item_idx, question, answer_a, answer_b in tasks:
            result = {
                "id": item_idx,
                "question": question,
                "answer1": answer_a,
                "answer2": answer_b
            }

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                response = future.result()
                result[f'judge_model_{key}_response'] = response[0]

            results.append(result)

        # 保存结果到JSON文件，两个项目一起保存
        with open(output_path, 'w') as outfile:
            json.dump(results, outfile, indent=4)
        print(f'问题ID {idx} 和 {idx+1} 和{idx+2}已评测并保存.')

print("已完成评测测试。")
