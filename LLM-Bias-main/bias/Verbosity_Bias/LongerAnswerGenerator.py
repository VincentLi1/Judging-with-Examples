import json
from promptTemplate import longSentence
from utils import get_openai_response

def process_data():
    try:
        # 尝试加载已处理的数据文件
        with open('resultsFormattedLonger.json', 'r') as file:
            data = json.load(file)
            print("Continuing from saved progress...")
    except FileNotFoundError:
        # 如果未处理的数据文件不存在，则从原始数据开始处理
        with open('data/resultsFormatted.json', 'r') as file:
            data = json.load(file)
            # 初始化id字段
            for entry in data:
                entry['id'] = data.index(entry)  # 添加ID

    # 检查每个条目是否已经有扩展答案
    for entry in data:
        if 'Answer2Longer' not in [response['model'] for response in entry['model_responses']]:  # 检查是否已有Answer2Longer
            answer2 = entry['model_responses'][1]['response'][0]
            extended_prompt = longSentence(answer2)
            extended_answer = get_openai_response("gpt-4", extended_prompt)
            
            # 新增一个Answer2Longer响应
            answer2_longer = {
                "model": "Answer2Longer",
                "response": [extended_answer]
            }
            entry['model_responses'].append(answer2_longer)  # 添加到model_responses数组

            # 实时保存数据到新文件
            with open('resultsFormattedLonger.json', 'w') as file:
                json.dump(data, file, indent=4)
            
            print(f"Processed entry ID {entry['id']}: {extended_answer}")

if __name__ == "__main__":
    process_data()
    print("All required Answer2 responses have been extended and updated in resultsFormattedLonger.json.")
