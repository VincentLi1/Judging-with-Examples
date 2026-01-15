# LLM-Bias Project README

## Directory Structure

```
LLM-Bias/
│
├── bias/
│   ├── Authority_bias/
│   │   ├── ...  # Code and files related to Authority Bias
│   │   └── result.log
│   ├── Bandwagon_effect_bias/
│   │   ├── ...  # Code and files related to Bandwagon Effect Bias
│   │   └── result.log
│   ├── compassion_fade_bias/
│   │   ├── ...  # Code and files related to Compassion Fade Bias
│   │   └── result.log
│   ├── CoT_Bias/
│   │   ├── ...  # Code and files related to CoT Bias
│   │   └── result.log
│   ├── Diversity_bias/
│   │   ├── ...  # Code and files related to Diversity Bias
│   │   └── result.log
│   ├── Dstraction_Bias/
│   │   ├── ...  # Code and files related to Distraction Bias
│   │   └── result.log
│   ├── FallacyOversightAnswer/
│   │   ├── ...  # Code and files related to Fallacy Oversight Bias
│   │   └── result.log
│   ├── position_bias/
│   │   ├── ...  # Code and files related to Position Bias
│   │   └── result.log
│   ├── Refinement_bias/
│   │   ├── ...  # Code and files related to Refinement Bias
│   │   └── result.log
│   ├── Self_enhancement_bias/
│   │   ├── ...  # Code and files related to Self-enhancement Bias
│   │   └── result.log
│   ├── Sentiment_Bias/
│   │   ├── ...  # Code and files related to Sentiment Bias
│   │   └── result.log
│   ├── Verbosity_Bias/
│   │   ├── ...  # Code and files related to Verbosity Bias
│   │   └── result.log
│   └── ...  # Ignored: random_test and Random_Bias folders
│
├── data/
│   └── ...  # Data folder, stores datasets
├── .gitignore
├── data_generate.py
├── promptTemplate.py
├── utils.py
└── README.md
```

## File Descriptions

### Root Directory Files

- `data_generate.py`: Responsible for randomly extracting data from datasets to form the initial dataset. All generated data will be stored in the `data` folder.
- `promptTemplate.py`: Implements all the prompt templates, defining various prompts for sending requests to the models.
- `utils.py`: Handles API communication and data post-processing. It includes functions to handle responses from various models:
  - `mixtral-8x22b`
  - `Qwen2-72b`
  - `llama3-70b`
  - `llama3-8b`
  - `mistral-7b`
  - `claude-3.5-sonnet`
  - `gpt-35-turbo`
  - `gpt-4`
  - `gpt-4o`
  - `GLM-4`
- `README.md`: This file, providing an overview and details about the project.

### `bias` Folder

Contains multiple subfolders, each corresponding to a specific type of bias. Each subfolder includes the following contents:

- Bias-specific code: Code for handling the particular type of bias.
- Intermediate files: Temporary files generated during the processing.
- `result.log`: Logs the final evaluation results for the specific bias.

### Newly added prompt helpers

The following `*(g.t).py` helpers were created to provide a uniform `generate_prompt` interface for
previously missing bias categories. Each file includes attribution at the top of the script:

- `bias/FallacyOversightAnswer/FallacyOversightAnswer(g.t).py`
- `bias/Random_Bias/Random_Bias(g.t).py`
- `bias/Refinement_bias/Refinement_bias(g.t).py`
- `bias/Sentiment_Bias/Sentiment_Bias(g.t).py`
- `bias/Verbosity_Bias/Verbosity_Bias(g.t).py`
- `bias/compassion_fade_bias/compassion_fade_bias(g.t).py`
- `bias/position_bias/position_bias(g.t).py`
- `bias/random_test/random_test(g.t).py`

## Results Analysis

The evaluation results for each bias test are recorded in the respective `result.log` files within each subfolder. These results can be used to analyze the performance of large language models under various bias conditions. Further analysis methods can be designed and implemented based on project requirements.
