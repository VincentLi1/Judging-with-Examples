# Judging with Examples: Improving LLM Judge Alignment on Nonverifiable Rewards via Prompting for Examples
Vincent Li, Yixin Liu (advisor), Arman Cohan (advisor)  
December 2025

Large language models (LLMs) are increasingly used as automated judges in evaluation pipelines for tasks such as short-answer grading, alignment assessment, and model-to-model comparison. However, LLM-as-a-judge systems are known to suffer from misalignment with human preferences. Prior work has proposed solutions including rubrics, reference answers, checklists, and agentic judging workflows, yet these approaches do not directly address a core underlying weakness: many judge models simply fail due to a lack of grounding in their evaluation process—they do not understand what constitutes a good response, and what constitutes a bad response.

In this work, we introduce **Judging with Examples**, a simple and model-agnostic prompting strategy designed to increase the semantic grounding of LLM judges. Before the judge evaluates the target student response, it is first asked to produce example responses corresponding to each rubric score. This example-generation step encourages the model to construct an internal representation of the rubric, the task, and the expected semantic distinctions. The resulting examples are then appended to the original judging prompt, yielding a more grounded assessment.

We evaluate our method on a subset of twenty difficult instances from the BiGGen Bench, selected specifically for their large disagreement between two judge models (Llama-3.1-8B-Instruct and Qwen-2.5-14B-Instruct). Applying **Judging with Examples** using the Llama-3.1-8B-Instruct model reduces mean absolute deviation from GPT-5.1 scores by **20.5%** on these challenging cases. Our findings indicate that grounding the judge through explicit example construction is an effective and computationally lightweight method for improving score reliability, particularly for complex tasks requiring deep understanding.

**Note:** In our GitHub repo, we include submodules from other works, such as:

* FLASK — [https://github.com/kaistAI/FLASK](https://github.com/kaistAI/FLASK)
* ReIFE — [https://arxiv.org/abs/2410.07069](https://arxiv.org/abs/2410.07069)
* llm_grader — [https://github.com/wenjing1170/llm_grader](https://github.com/wenjing1170/llm_grader)
* prometheus-eval — [https://github.com/prometheus-eval/prometheus-eval](https://github.com/prometheus-eval/prometheus-eval)


