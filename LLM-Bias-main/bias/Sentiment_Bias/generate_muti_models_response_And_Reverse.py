import json
from promptTemplate import evaluate_ai_responses_no_tie
from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

judge_models = ['gpt-35-turbo', 'gpt-4', 'gpt-4o', 'GLM-4', 'claude-3.5-sonnet', 'Qwen2-72b']

# 读取问题和答案
with open('bias/Sentiment_Bias/resultsFormattedSentiment.json', 'r') as file:
    data = json.load(file)

output_path = 'bias/Sentiment_Bias/Sentiment_evaluationResults_FearAnswer2.json'
reverse_output_path = 'bias/Sentiment_Bias/Reverse_Sentiment_evaluationResults_FearAnswer2.json'

# 为正向和反向评测分别初始化结果列表
results = []
reverse_results = []

# 尝试从现有文件加载已有的结果
try:
    with open(output_path, 'r') as file:
        results = json.load(file)
    with open(reverse_output_path, 'r') as file:
        reverse_results = json.load(file)
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
    for idx in range(start_id, len(data), 5):
        tasks = []
        for sub_idx in [0, 1, 2, 3, 4]:
            if idx + sub_idx >= len(data):
                continue
            item = data[idx + sub_idx]
            if 'model_responses' not in item or len(item['model_responses']) < 2:
                continue

            question = item["question"]
            answer_a = item["model_responses"][0]['response'][0]
            answer_b = item["model_responses"][9]['response'][0]
            evaluate_prompt = evaluate_ai_responses_no_tie(question, answer_a, answer_b)
            reverse_evaluate_prompt = evaluate_ai_responses_no_tie(question, answer_b, answer_a)  # 反向提示

            future_to_key = {executor.submit(evaluate_model, key, evaluate_prompt): key for key in judge_models}
            reverse_future_to_key = {executor.submit(evaluate_model, key, reverse_evaluate_prompt): key for key in judge_models}
            tasks.append((future_to_key, reverse_future_to_key, idx + sub_idx, question, answer_a, answer_b))
        
        for future_to_key, reverse_future_to_key, item_idx, question, answer_a, answer_b in tasks:
            result = {
                "id": item_idx,
                "question": question,
                "answer1": answer_a,
                "answer2": answer_b
            }
            reverse_result = {
                "id": item_idx,
                "question": question,
                "answer1": answer_b,
                "answer2": answer_a
            }

            for future, reverse_future in zip(as_completed(future_to_key), as_completed(reverse_future_to_key)):
                key = future_to_key[future]
                reverse_key = reverse_future_to_key[reverse_future]
                response = future.result()
                reverse_response = reverse_future.result()
                result[f'judge_model_{key}_response'] = response[0]
                reverse_result[f'judge_model_{reverse_key}_response'] = reverse_response[0]

            results.append(result)
            reverse_results.append(reverse_result)

        # 保存结果到JSON文件，两个项目一起保存
        with open(output_path, 'w') as outfile:
            json.dump(results, outfile, indent=4)
        with open(reverse_output_path, 'w') as reverse_outfile:
            json.dump(reverse_results, reverse_outfile, indent=4)
        print(f'问题ID {idx} 和 {idx+1} 和{idx+2}和{idx+3}和{idx+4}已评测并保存.')

print("已完成评测测试。")
