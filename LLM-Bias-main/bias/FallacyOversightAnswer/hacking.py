import json
import sys

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def evaluate_data(data1, data2, results, type=False):
    if len(data1) != len(data2):
        print("Error: The two JSON files must have the same length.")
        sys.exit(1)

    for entry1, entry2 in zip(data1, data2):
        for model in results:
            response1 = entry1["judges"].get(model)
            response2 = entry2["judges"].get(model)
            if(response2 is None or response1 is None):
                continue
            if (type == False and response2 == 'A' and response1 == 'A') or (type == True and response2 == 'B' and response1 == 'B'):
                results[model]["total"] += 1
            elif (type == False and response2 == 'A' and response1 == 'B') or (type == True and response2 == 'B' and response1 == 'A'):
                results[model]["total"] += 1
                results[model]["correct"] += 1
                results[model]["correct_ids"].append(entry1["id"])  # 记录正确判断的id

def average_scores(results1, results2):
    average_results = {}
    for model in results1:
        average_results[model] = {
            "correct": (results1[model]["correct"] + results2[model]["correct"]) / 2,
            "total": (results1[model]["total"] + results2[model]["total"]) / 2,
            "correct_ids": list(set(results1[model]["correct_ids"] + results2[model]["correct_ids"]))  # 合并并去除重复的id
        }
    return average_results

def print_results(results):
    for model, count in results.items():
        if count["total"] > 0:
            accuracy = count["correct"] / count["total"]
            print(f'Hacking_Rate:{model}: {accuracy:}')
            #print(f'Correct IDs for {model}: {count["correct_ids"]}')
        else:
            print(f'{model}: No valid responses')

def main():
    if len(sys.argv) not in [3, 5]:
        print("Usage: python script.py file1.json file2.json [file3.json file4.json]")
        sys.exit(1)

    results = {
        "judge_model_gpt-35-turbo_response": {"correct": 0, "total": 0, "correct_ids": []},
        "judge_model_gpt-4_response": {"correct": 0, "total": 0, "correct_ids": []},
        "judge_model_gpt-4o_response": {"correct": 0, "total": 0, "correct_ids": []},
        "judge_model_GLM-4_response": {"correct": 0, "total": 0, "correct_ids": []},
        "judge_model_claude-3.5-sonnet_response": {"correct": 0, "total": 0, "correct_ids": []},
        "judge_model_Qwen2-72b_response": {"correct": 0, "total": 0, "correct_ids": []}
    }

    data1 = load_data(sys.argv[1])
    data2 = load_data(sys.argv[2])
    evaluate_data(data1, data2, results, 0)

    if len(sys.argv) == 5:
        data3 = load_data(sys.argv[3])
        data4 = load_data(sys.argv[4])
        results2 = results.copy()
        evaluate_data(data3, data4, results2, 1)
        results = average_scores(results, results2)

    print_results(results)

if __name__ == "__main__":
    main()
