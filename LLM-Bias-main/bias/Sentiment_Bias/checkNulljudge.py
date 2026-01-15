import json
from promptTemplate import evaluate_ai_responses_no_tie
from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

judge_models = ['gpt-35-turbo', 'gpt-4', 'gpt-4o', 'GLM-4', 'claude-3.5-sonnet', 'Qwen2-72b']

output_path = 'data/evaluationResultsLonger.json'

# 读取和检查已有的结果
with open(output_path, 'r') as file:
    results = json.load(file)

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
    for result in results:
        question = result["question"]
        answer_a = result["answer1"]
        answer_b = result["answer2"]
        evaluate_prompt = evaluate_ai_responses_no_tie(question, answer_a, answer_b)

        # 检查每个模型的响应并重新评估null结果
        reevaluated = False
        future_to_key = {}
        for key in judge_models:
            response_key = f'judge_model_{key}_response'
            if result.get(response_key) is None:
                # 重新提交评估
                future_to_key[executor.submit(evaluate_model, key, evaluate_prompt)] = key
                reevaluated = True

        # 更新任何重新评估的响应
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            response = future.result()
            result[f'judge_model_{key}_response'] = response[0] if response else 'Error: Response was null'

        # 如果进行了重新评估，则输出提示
        if reevaluated:
            print(f"问题ID {result['id']} 有重新评估并更新的结果。")

# 保存更新后的结果到JSON文件
with open(output_path, 'w') as outfile:
    json.dump(results, outfile, indent=4)

print("检查并更新完成。")
