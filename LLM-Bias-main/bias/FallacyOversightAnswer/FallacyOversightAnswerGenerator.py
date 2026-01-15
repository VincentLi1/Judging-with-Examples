import json
import concurrent.futures
from promptTemplate import longSentence, getFallacyOversightAnswer
from utils import get_openai_response

def process_entry(entry):
    if 'Answer1FallacyOversight' not in [response['model'] for response in entry['model_responses']]:
        answer1 = entry['model_responses'][0]['response'][0]
        extended_prompt = getFallacyOversightAnswer(answer1)
        extended_answer = get_openai_response("gpt-4", extended_prompt)

        Answer1FallacyOversight = {
            "model": "Answer1FallacyOversight",
            "response": [extended_answer]
        }
        return (entry['id'], Answer1FallacyOversight)
    return None

def save_to_file(data, updates):
    for entry_id, update in updates.items():
        data[entry_id]['model_responses'].append(update)
    with open('resultsFormattedFallacyOversight.json', 'w') as file:
        json.dump(data, file, indent=4)

def process_data():
    try:
        with open('resultsFormattedFallacyOversight.json', 'r') as file:
            data = json.load(file)
            print("Continuing from saved progress...")
    except FileNotFoundError:
        with open('data/resultsFormatted.json', 'r') as file:
            data = json.load(file)
            for entry in data:
                entry['id'] = data.index(entry)

    updates = {}
    batch_size = 4
    # Create a thread pool to process entries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Process entries in batches of 4
        for i in range(0, len(data), batch_size):
            futures = [executor.submit(process_entry, data[j]) for j in range(i, min(i + batch_size, len(data)))]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    entry_id, update = result
                    updates[entry_id] = update

            # Save and print every batch of 4
            if updates:
                save_to_file(data, updates)
                for entry_id in updates:
                    print(f"Processed entry ID {entry_id}: {updates[entry_id]['response'][0]}")
                updates = {}  # Reset updates after saving

if __name__ == "__main__":
    process_data()
    print("All required Answer1 responses have been extended and updated in resultsFormattedFallacyOversight.json.")
