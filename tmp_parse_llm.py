import json
from pathlib import Path
from parsers import simple_pointwise_parse
from data_models import JudgePrediction
path = Path('results/llm_grader/llm_grader_pointwise_vanilla.base_pointwise.qwen2.5-0.5b.original.jsonl')
fails = []
with path.open() as f:
    for line in f:
        pred = JudgePrediction.from_dict(json.loads(line))
        _, fail = simple_pointwise_parse(pred, verbose=False)
        if fail:
            fails.append((pred.id, pred.response[0]['text'] if pred.response else ''))
            if len(fails) >= 5:
                break
print('total fails captured', len(fails))
for idx, (id_, resp) in enumerate(fails, 1):
    print(f"\nExample {idx} id={id_}\nResponse: {resp}")
