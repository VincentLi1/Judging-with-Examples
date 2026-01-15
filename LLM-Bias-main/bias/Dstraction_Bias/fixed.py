import json

# Load your original JSON file
with open('Modified_Dstraction_Bias(g.t)_A.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Given indices where responses are missing for each model
missing_indices = [
    [],
    [],
    [],
    [],
    [299, 301, 304, 323],
    []
]

# Function to insert null at specified missing indices in responses list
def insert_missing_responses(responses, missing_ids):
    for index in sorted(missing_ids, reverse=True):
        responses.insert(index, None)
    return responses

# Update the responses in the JSON data
for i, model_data in enumerate(data):
    if 'responses' in model_data:
        print(f"Updating model {i} with responses available.")
        model_data['responses'] = insert_missing_responses(model_data['responses'], missing_indices[i])
    else:
        print(f"No 'responses' key found in model {i}:")

# Save the modified JSON to a new file
with open('Modified_Dstraction_Bias(g.t)_AA.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
