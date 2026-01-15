import json
import sys

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def evaluate_data(data, results, score_criteria):
    for entry in data:
        for model, response in entry["judges"].items():
            if response is not None:
                results[model]["total"] += 1
            if response == score_criteria:
                results[model]["correct"] += 1

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py [single/double/reverse] [1.json] [2.json (if double)]")
        sys.exit(1)

    mode = sys.argv[1]
    file1_path = sys.argv[2]
    file2_path = sys.argv[3] if len(sys.argv) > 3 else None

    if mode == "double" and not file2_path:
        print("Error: Double mode requires two JSON files.")
        sys.exit(1)

    # 初始化计数器
    results = {
        "judge_model_gpt-35-turbo_response": {"correct": 0, "total": 0},
        "judge_model_gpt-4_response": {"correct": 0, "total": 0},
        "judge_model_gpt-4o_response": {"correct": 0, "total": 0},
        "judge_model_GLM-4_response": {"correct": 0, "total": 0},
        "judge_model_claude-3.5-sonnet_response": {"correct": 0, "total": 0},
        "judge_model_Qwen2-72b_response": {"correct": 0, "total": 0}
    }

    # 处理第一个文件
    if(mode!="reverse"):
        data1 = load_data(file1_path)
        evaluate_data(data1, results, "A")
    else:
        data1 = load_data(file1_path)
        evaluate_data(data1, results, "B")
    if mode == "double":
        # 处理第二个文件
        data2 = load_data(file2_path)
        evaluate_data(data2, results, "B")

    # 计算并输出正确率
    for model, count in results.items():
        if count["total"] > 0:
            accuracy = count["correct"] / count["total"]
            print(f'Accuracy:{model}: {accuracy:}')
        else:
            print(f'{model}: No valid responses')

if __name__ == "__main__":
    main()
