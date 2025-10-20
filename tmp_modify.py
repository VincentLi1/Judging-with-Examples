from pathlib import Path
text = Path('ReIFE/ReIFE/base_llm.py').read_text()
needle = "        outputs = self.model.generate(\n"
start = text.index(needle)
end = text.index("        _outputs = []", start)
new_block = "        sampling_params = SamplingParams(\n            n=n,\n            temperature=temperature,\n            max_tokens=max_tokens,\n            top_p=top_p,\n            logprobs=logprobs,\n            stop_token_ids=self.STOP_TOKEN_IDS,\n        )\n\n        try:\n            outputs = self.model.generate(\n                prompt_token_ids=prompts,\n                sampling_params=sampling_params,\n                use_tqdm=use_tqdm,\n            )\n        except TypeError as exc:\n            if \"prompt_token_ids\" not in str(exc):\n                raise\n            prompt_texts = self.tokenizer.batch_decode(\n                prompts, skip_special_tokens=False, clean_up_tokenization_spaces=False\n            )\n            outputs = self.model.generate(\n                prompts=prompt_texts,\n                sampling_params=sampling_params,\n                use_tqdm=use_tqdm,\n            )\n"
text = text[:start] + new_block + text[end:]
Path('ReIFE/ReIFE/base_llm.py').write_text(text)
