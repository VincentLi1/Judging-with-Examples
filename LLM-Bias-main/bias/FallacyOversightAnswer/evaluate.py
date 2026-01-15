import json

def process_evaluation_results(input_filename, output_filename):
    # 读取JSON文件
    with open(input_filename, 'r') as file:
        data = json.load(file)
    
    # 用于存储处理后的结果
    processed_results = []

    # 遍历每一个问题的数据
    for item in data:
        # 提取问题ID
        question_id = item['id']
        # 准备存储评价结果
        judges = {}

        # 遍历每一个模型的评价结果
        for key, value in item.items():
            if key.startswith('judge_model_'):
                if value is None:
                    judges[key] = None
                else:
                    # 获取评价文本
                    response_text = value[0]  # 确保value不是None才进行访问
                    # 确定评价结果
                    result = None
                    if '[[A]]' in response_text and '[[B]]' not in response_text:
                        result = 'A'
                    elif '[[B]]' in response_text and '[[A]]' not in response_text:
                        result = 'B'
                    # 存储结果
                    judges[key] = result
        
        # 添加到处理后的结果列表
        processed_results.append({
            "id": question_id,
            "judges": judges
        })
    
    # 将处理后的结果保存到新的JSON文件
    with open(output_filename, 'w') as f:
        json.dump(processed_results, f, indent=4)
# 你可以通过调用这个函数来处理你的文件
process_evaluation_results('Reverse_FallacyOversight_evaluationResults.json', '2Reverse_FallacyOversight_evaluationResults.json')
