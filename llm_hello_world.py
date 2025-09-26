import sys
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def require_gpu():
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available.", file=sys.stderr)
        sys.exit(2)


def generate_hello(model_id: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True,
    )

    # Prefer float16 on GPU for small models
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Basic chat messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello world"},
    ]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        # Fallback if no chat template
        prompt = "\n".join(m["content"] for m in messages)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Move to the model device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None and tokenizer.eos_token_id is not None:
        pad_id = tokenizer.eos_token_id

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=pad_id
        )

    gen_ids = output[0, input_ids.shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main() -> int:
    require_gpu()
    candidates = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]

    failures: list[tuple[str, str]] = []
    for model_id in candidates:
        print(f"Loading model: {model_id}")
        try:
            answer = generate_hello(model_id)
            print("Answer:")
            print(answer)
            return 0
        except Exception as e:
            failures.append((model_id, str(e)))
            print(f"Failed with {model_id}: {e}", file=sys.stderr)
            traceback.print_exc()
            print("Trying next model...\n", file=sys.stderr)

    print("All candidates failed:")
    for mid, err in failures:
        print(f"- {mid}: {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
