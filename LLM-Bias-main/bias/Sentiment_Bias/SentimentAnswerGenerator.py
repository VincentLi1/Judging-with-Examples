import json
import concurrent.futures
import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取需要导入的模块的目录
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

# 将模块的目录添加到sys.path中
sys.path.append(parent_dir)
from promptTemplate import *
from utils import get_openai_response

def process_entry(entry):
    model_responses = entry['model_responses']
    response_models = [response['model'] for response in model_responses]

    if 'Answer1Cheerful' not in response_models:
        answer1 = model_responses[0]['response'][0]
        extended_prompt = getCheerfulAnswer(answer1)
        extended_answer = get_openai_response("gpt-4", extended_prompt)

        model_responses.append({
            "model": "Answer1Cheerful",
            "response": [extended_answer]
        })

    if 'Answer1Sad' not in response_models:
        answer1 = model_responses[0]['response'][0]
        extended_prompt = getSadAnswer(answer1)
        extended_answer = get_openai_response("gpt-4", extended_prompt)

        model_responses.append({
            "model": "Answer1Sad",
            "response": [extended_answer]
        })

    if 'Answer2Cheerful' not in response_models:
        answer2 = model_responses[1]['response'][0]
        extended_prompt = getCheerfulAnswer(answer2)
        extended_answer = get_openai_response("gpt-4", extended_prompt)

        model_responses.append({
            "model": "Answer2Cheerful",
            "response": [extended_answer]
        })

    if 'Answer2Sad' not in response_models:
        answer2 = model_responses[1]['response'][0]
        extended_prompt = getSadAnswer(answer2)
        extended_answer = get_openai_response("gpt-4", extended_prompt)

        model_responses.append({
            "model": "Answer2Sad",
            "response": [extended_answer]
        })

    if 'Answer1Angry' not in response_models:
        answer1 = model_responses[0]['response'][0]
        extended_prompt = getAngryAnswer(answer1)
        extended_answer = get_openai_response("gpt-4", extended_prompt)

        model_responses.append({
            "model": "Answer1Angry",
            "response": [extended_answer]
        })

    if 'Answer1Fear' not in response_models:
        answer1 = model_responses[0]['response'][0]
        extended_prompt = getFearAnswer(answer1)
        extended_answer = get_openai_response("gpt-4", extended_prompt)

        model_responses.append({
            "model": "Answer1Fear",
            "response": [extended_answer]
        })

    if 'Answer2Angry' not in response_models:
        answer2 = model_responses[1]['response'][0]
        extended_prompt = getAngryAnswer(answer2)
        extended_answer = get_openai_response("gpt-4", extended_prompt)

        model_responses.append({
            "model": "Answer2Angry",
            "response": [extended_answer]
        })

    if 'Answer2Fear' not in response_models:
        answer2 = model_responses[1]['response'][0]
        extended_prompt = getFearAnswer(answer2)
        extended_answer = get_openai_response("gpt-4", extended_prompt)

        model_responses.append({
            "model": "Answer2Fear",
            "response": [extended_answer]
        })

    return model_responses


def save_to_file(data):
    with open('resultsFormattedSentiment.json', 'w') as file:
        json.dump(data, file, indent=4)


def process_data():
    try:
        with open('resultsFormattedSentiment.json', 'r') as file:
            data = json.load(file)
            print("Continuing from saved progress...")
    except FileNotFoundError:
        with open('data/selectedResultsFormatted.json', 'r') as file:
            data = json.load(file)
            for entry in data:
                entry['id'] = data.index(entry)

    batch_size = 4
    # Create a thread pool to process entries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(data), batch_size):
            futures = [executor.submit(process_entry, data[j]) for j in range(i, min(i + batch_size, len(data)))]
            for future in futures:
                result = future.result()
                if result:
                    entry_id = result[0]  # Assuming the first element of result is the entry_id
                    updates = result[1]  # Assuming the second element of result is the updates dictionary

                    # Find the correct entry to update by entry_id
                    for entry in data:
                        if entry['id'] == entry_id:
                            entry['model_responses'] = updates
                            break
                    print(f"Processed entry ID {entry_id}.")

            # Save the updated data after processing each batch
            save_to_file(data)
            print(f"Saved progress up to entry ID {data[min(i + batch_size - 1, len(data) - 1)]['id']}.")


if __name__ == "__main__":
    process_data()
    print("All required Answer1 responses have been extended and updated in resultsFormattedFallacyOversight.json.")
