import json
from promptTemplate import generate_polish_prompt, get_score
from utils import * 
from concurrent.futures import ThreadPoolExecutor, as_completed

models = ['gpt-35-turbo', 'gpt-4', 'gpt-4o', 'GLM-4', 'claude-3.5-sonnet', 'Qwen2-72b']

with open('questions.json', 'r') as file:
    questions = json.load(file)

output_path = 'answers_with_polish_and_scores.json'

try:
    with open(output_path, 'r') as file:
        results = json.load(file)
        start_id = len(results)
except FileNotFoundError:
    results = []
start_id = 0

def get_model_response(model, prompt):
    if model in ['gpt-35-turbo', 'gpt-4', 'gpt-4o']:
        return get_multiple_openai_responses([(model, prompt)])
    elif model in ['GLM-4', 'claude-3.5-sonnet']:
        return get_multiple_other_model_responses([(model, prompt)])
    elif model in ["mixtral-8x22b", "Qwen2-72b", "llama3-70b"]:
        return get_multiple_large_model_responses([(model, prompt)])
    elif model in ["llama3-8b"]:
        return get_multiple_opensource_model_responses([(model, prompt)])
    else:
        return None

def process_question_for_model(model, question, initial_prompt):
# Get initial answer
    initial_response = get_model_response(model, initial_prompt)
    if not initial_response:
        return None, None, None, None
    initial_answer = initial_response[0]

    # Polish the answer
    history = f"user: {question}\n{model}: {initial_answer}"
    polish_prompt = generate_polish_prompt(history)
    polished_response = get_model_response(model, polish_prompt)

    if not polished_response:
        return initial_answer, None, None, None, None

    polished_answer = polished_response[0]

    # Score the polished answer
    score_prompt_0=get_score(question, initial_answer)
    score_response_0=get_model_response(model, score_prompt_0)
    score_prompt_1 = get_score(question, polished_answer)
    score_response_1 = get_model_response(model, score_prompt_1)
    
    # Score the full conversation
    full_conversation = f"Initial answer: {initial_answer}\nUser: Please polish this answer to make it better and more complete.\nPolished answer: {polished_answer}"
    score_prompt_2 = get_score(question, full_conversation)
    score_response_2 = get_model_response(model, score_prompt_2)

    return initial_answer, polished_answer, score_response_0[0] if score_response_0 else None, score_response_1[0] if score_response_1 else None, score_response_2[0] if score_response_2 else None

with ThreadPoolExecutor(max_workers=len(models)) as executor:
    for idx in range(start_id, len(questions), 5):
        tasks = []
        for sub_idx in [0, 1, 2, 3, 4]:
            if idx + sub_idx >= len(questions):
                continue
            question = questions[idx + sub_idx]['question']
            initial_prompt = question

            future_to_model = {executor.submit(process_question_for_model, model, question, initial_prompt): model for model in models}
            tasks.append((future_to_model, idx + sub_idx, question))

        for future_to_model, item_idx, question in tasks:
            result = {
                "id": item_idx,
                "question": question,
                "FROM": questions[item_idx]['FROM']
            }

            for future in as_completed(future_to_model):
                model = future_to_model[future]
                initial_answer, polished_answer, score_0, score_1, score_2 = future.result()
                if initial_answer:
                    result[f'{model}_initial_response'] = initial_answer
                    result[f'{model}_polished_response'] = polished_answer
                    result[f'{model}_initial_score'] = score_0
                    result[f'{model}_polished_score'] = score_1
                    result[f'{model}_full_conversation_score'] = score_2

            results.append(result)

        # 保存结果到JSON文件
        with open(output_path, 'w') as outfile:
            json.dump(results, outfile, indent=4)
        print(f'问题ID {idx}, {idx+1}, {idx+2}, {idx+3},{idx+4}已处理并保存.')

print("所有问题已处理完毕。")

print("All questions have been processed with initial answers, polished answers, and scores.")