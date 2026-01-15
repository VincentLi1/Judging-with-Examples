import json
import sys

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def evaluate_data(data1, data2, results):
    if len(data1) != len(data2):
        print("Error: The two JSON files must have the same length.")
        print(len(data1))
        print(len(data2))
        sys.exit(1)

    for entry1, entry2 in zip(data1, data2):
        for model in results:
            response1 = entry1["judges"].get(model)
            response2 = entry2["judges"].get(model)
            if response1 == response2 and (response1 == "A" or response1 == "B"):
                results[model]["total"] += 1
            # elif (response1 is None and response2 is not None) or (response1 is not None and response2 is None):
            #     results[model]["correct"] += 1
            #     results[model]["total"] += 1
            elif response1 is not None and response2 is not None and response1 != response2:
                results[model]["total"] += 1
                results[model]["correct"] += 1

def average_scores(results1, results2):
    average_results = {}
    for model in results1:
        average_results[model] = {
            "correct": (results1[model]["correct"] + results2[model]["correct"]) / 2,
            "total": (results1[model]["total"] + results2[model]["total"]) / 2
        }
    return average_results

def print_results(results):
    for model, count in results.items():
        if count["total"] > 0:
            accuracy = count["correct"] / count["total"]
            print(f'Consistency:{model}: {1-accuracy:}')
        else:
            print(f'{model}: No valid responses')

def main():
    if len(sys.argv) not in [3, 5]:
        print("Usage: python script.py file1.json file2.json [file3.json file4.json]")
        sys.exit(1)

    results = {
        "judge_model_gpt-35-turbo_response": {"correct": 0, "total": 0},
        "judge_model_gpt-4_response": {"correct": 0, "total": 0},
        "judge_model_gpt-4o_response": {"correct": 0, "total": 0},
        "judge_model_GLM-4_response": {"correct": 0, "total": 0},
        "judge_model_claude-3.5-sonnet_response": {"correct": 0, "total": 0},
        "judge_model_Qwen2-72b_response": {"correct": 0, "total": 0}
    }

    data1 = load_data(sys.argv[1])
    data2 = load_data(sys.argv[2])
    evaluate_data(data1, data2, results)

    if len(sys.argv) == 5:
        data3 = load_data(sys.argv[3])
        data4 = load_data(sys.argv[4])
        results2 = results.copy()
        evaluate_data(data3, data4, results2)
        results = average_scores(results, results2)

    print_results(results)

if __name__ == "__main__":
    main()
