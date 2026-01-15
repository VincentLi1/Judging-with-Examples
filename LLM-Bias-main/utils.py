import os
import re
import replicate
import replicate.client
import anthropic
from functools import wraps
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union
from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI
from dotenv import load_dotenv
model_dict = {
    "mixtral-8x22b" : "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "Qwen2-72b" : "Qwen/Qwen2-72B-Instruct",
    "llama3-70b" : "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3-8b": "meta/meta-llama-3-8b-instruct",
    "mistral-7b" : "mistralai/mistral-7b-instruct-v0.2",
    "claude-3.5-sonnet" : "claude-3-5-sonnet-20240620",
    "gpt-35-turbo": "yuehuang-chatgpt",
    "gpt-4" : "yuehuang-gpt-4", 
    "gpt-4o" : "yuehuang-gpt-4o",
    "GLM-4" : "glm-4"
}
judge_model_set = ['gpt-35-turbo', 'gpt-4', 'gpt-4o', 'GLM-4', 'claude-3.5-sonnet', 'Qwen2-72b']
generate_model_set = ['mixtral-8x22b', 'llama3-70b', 'llama3-8b', 'mistral-7b']
load_dotenv()
for key in ["REPLICATE_API_TOKEN", "https_proxy", "http_proxy"]:
    val = os.getenv(key)
    if val is not None:
        os.environ[key] = val

# ---------------------------------------------------------------------------
# Stubs to disable external API calls (offline mode)
# ---------------------------------------------------------------------------
def _stub_response(prompts):
    # Always return a list of placeholder strings matching input length.
    return ["API not available"] * len(prompts)

# Expose a simple stub for any direct call pattern used elsewhere.
def get_openai_response(*args, **kwargs):
    return "API not available"


# Expose a simple stub for replicate-style calls.
def get_replicate_response(*args, **kwargs):
    return "API not available"

# Ensure any lookups via model_function use the stub instead of real APIs.

class TokenLogger:
    def __init__(self, filename="token_log.txt"):
        self.filename = filename
        self.model_tokens = {}
        self.load_tokens()

    def log_tokens(self, model_name, input_tokens, output_tokens):
        if model_name not in self.model_tokens:
            self.model_tokens[model_name] = {"input_total": 0, "output_total": 0, "last_input": 0, "last_output": 0}
        
        self.model_tokens[model_name]["input_total"] += input_tokens
        self.model_tokens[model_name]["output_total"] += output_tokens
        self.model_tokens[model_name]["last_input"] = input_tokens
        self.model_tokens[model_name]["last_output"] = output_tokens

        print(f"Model: {model_name}, Input tokens: {input_tokens}, Output tokens: {output_tokens}, "
              f"Total input tokens: {self.model_tokens[model_name]['input_total']}, "
              f"Total output tokens: {self.model_tokens[model_name]['output_total']}")

        self.save_tokens()

    def get_total_tokens(self, model_name):
        if model_name in self.model_tokens:
            return self.model_tokens[model_name]["input_total"], self.model_tokens[model_name]["output_total"]
        else:
            return 0, 0

    def save_tokens(self):
        with open(self.filename, 'w') as file:
            for model_name, tokens in self.model_tokens.items():
                file.write(f"{model_name},{tokens['input_total']},{tokens['output_total']},{tokens['last_input']},{tokens['last_output']}\n")

    def load_tokens(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as file:
                for line in file:
                    model_name, input_total, output_total, last_input, last_output = line.strip().split(',')
                    self.model_tokens[model_name] = {
                        "input_total": int(input_total),
                        "output_total": int(output_total),
                        "last_input": int(last_input),
                        "last_output": int(last_output)
                    }
        else:
            self.model_tokens = {}

def count_tokens(text):
    """
    Counts the number of tokens in the given text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of tokens in the text.
    """
    if not isinstance(text, str):
        text = str(text)
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return len(tokens) * 4 // 3

logger = TokenLogger()

def token_logger_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        model_name = args[0] if len(args) > 0 else kwargs.get('model_name', '')
        prompt = args[1] if len(args) > 1 else kwargs.get('prompt', '')
        
        input_tokens = count_tokens(prompt)
        response, output_tokens = func(*args, **kwargs)
        logger.log_tokens(model_name, input_tokens, output_tokens)
        return response, output_tokens
    return wrapper

@token_logger_decorator
def get_opensource_model_response(model_name, prompt, temperature=0.6, max_tokens=1024):
    """
    Retrieves the response from an open-source model given a prompt.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The prompt to provide to the model.
        temperature (float, optional): The temperature parameter for generating the response. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        tuple: A tuple containing the generated response as a string and the number of tokens in the response.
    """
    
    input = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    }

    output = replicate.run(
        ref=model_dict[model_name],
        input=input,
    )
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens("".join(output))
    return "".join(output), count_tokens("".join(output))

@token_logger_decorator
def get_large_model_response(model_name, prompt, temperature=0.6, max_tokens=1024):
    """
    Generates a response from a large model using OpenAI's chat completions API.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The user's prompt or message.
        temperature (float, optional): The temperature parameter for generating the response. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        tuple: A tuple containing the generated response text and the number of tokens in the response.
    """
    openai = OpenAI(
        api_key=os.getenv("DEEPINFRA_TOKEN"),
        base_url="https://api.deepinfra.com/v1/openai",
    )
    stream = False
    chat_completion = openai.chat.completions.create(
        model=model_dict[model_name],
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )
    response_text = chat_completion.choices[0].message.content
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(response_text)
    return response_text, count_tokens(response_text)

@token_logger_decorator
def get_openai_response(model_name, prompt, temperature=0.6, max_tokens=1024):
    """
    Retrieves a response from the OpenAI API using the specified model.

    Args:
        model_name (str): The name of the OpenAI model to use.
        prompt (str or list): The prompt or list of prompts to generate a response for.
        temperature (float, optional): Controls the randomness of the generated text. Higher values result in more random outputs. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        tuple: A tuple containing the generated response text and the number of tokens in the response.
    """
   
    if model_name == "gpt-4o":
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_GPT4O_API_KEY"),
            api_version="2024-05-01-preview",
            azure_endpoint=os.getenv("AZURE_GPT4O_ENDPOINT")
        )
    else:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    model = model_dict[model_name]
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    # Request a completion from the API
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    response_text = response.choices[0].message.content
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(response_text)
    return response_text, count_tokens(response_text)

@token_logger_decorator
def get_other_model_response(model_name, prompt, temperature=0.6, max_tokens=1024):
    """
    Get the response from a specific model using the ZhipuAI chat completions API.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The prompt for the model to generate a response to.
        temperature (float, optional): The temperature parameter for generating the response. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        tuple: A tuple containing the generated response text and the number of tokens in the response.
    """
    if model_name == "GLM-4":
        client = ZhipuAI(
            api_key=os.getenv("ZHIPU_API_KEY"),
        )
        response = client.chat.completions.create(
            model=model_dict[model_name],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        response_text = response.choices[0].message.content
        input_tokens = count_tokens(prompt)
        output_tokens = count_tokens(response_text)
        return response_text, count_tokens(response_text)
    elif model_name == "claude-3.5-sonnet":
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        message = client.messages.create(
            model = model_dict[model_name],
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text
        input_tokens = count_tokens(prompt)
        output_tokens = count_tokens(response_text)
        return response_text, count_tokens(response_text)

async def aget_openai_response(model_name, prompt, temperature=0.6, max_tokens=1024):
    """
    Asynchronously sends a prompt to the OpenAI API and retrieves the response.

    Args:
        model_name (str): The name of the OpenAI model to use.
        prompt (str or list): The prompt to send to the API. If a string, it will be converted to a list with a single message.
        temperature (float, optional): The temperature parameter for generating the response. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        tuple: A tuple containing the response text and the number of tokens in the response.
    """
    
    if model_name == "gpt-4o":
        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_GPT4O_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_GPT4O_ENDPOINT")
        )
    else:
        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    model = model_dict[model_name]
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    # Request a completion from the API
    response = await client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    response_text = response.choices[0].message.content
    await client.close()
    return response_text, count_tokens(response_text)

async def aget_opensource_model_response(model_name, prompt, temperature=0.6, max_tokens=1024):
    """
    Asynchronously retrieves the response from an open-source model given a prompt.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The prompt to provide to the model.
        temperature (float, optional): The temperature parameter for generating the response. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        tuple: A tuple containing the generated response as a string and the number of tokens in the response.
    """
    input = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_template": "<s>[INST] {prompt} [/INST] "
    }
    output = await replicate.async_run(
        ref=model_dict[model_name],
        input=input,
    )
    
    return "".join(output), count_tokens("".join(output))

async def aget_large_model_response(model_name, prompt, temperature=0.6, max_tokens=1024):
    """
    Asynchronously generates a response from a large model using OpenAI's chat completions API.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The user's prompt or message.
        temperature (float, optional): The temperature parameter for generating the response. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        tuple: A tuple containing the generated response text and the number of tokens in the response.
    """
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPINFRA_TOKEN"),
        base_url="https://api.deepinfra.com/v1/openai",
    )
    stream = False
    chat_completion = await client.chat.completions.create(
        model=model_dict[model_name],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )
    response_text = chat_completion.choices[0].message.content
    return response_text, count_tokens(response_text)

async def aget_other_model_response(model_name, prompt, temperature=0.6, max_tokens=1024):
    """
    Asynchronously retrieves the response from a specific model using the ZhipuAI chat completions API.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The prompt for the model to generate a response to.
        temperature (float, optional): The temperature parameter for generating the response. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        tuple: A tuple containing the generated response text and the number of tokens in the response.
    """
    if model_name == "GLM-4":
        client = AsyncOpenAI(
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
        stream = False
        chat_completion = await client.chat.completions.create(
            model=model_dict[model_name],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        response_text = chat_completion.choices[0].message.content
        return response_text, count_tokens(response_text)
    
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union

def get_multiple_openai_responses(prompts: List[Tuple[str, Union[str, List[dict]]]], temperature: float = 0.6, max_tokens: int = 1024) -> List[Tuple[str, int]]:
    """
    Retrieves multiple responses from the OpenAI API using the specified model.

    Args:
        prompts (list): A list of tuples, where each tuple contains the model name and the prompt.
        temperature (float, optional): Controls the randomness of the generated text. Higher values result in more random outputs. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        list: A list of tuples, each containing the generated response text and the number of tokens in the response.
    """
    responses = [None] * len(prompts)
    
    def task_wrapper(index, model_name, prompt):
        response = get_openai_response(model_name, prompt, temperature, max_tokens)[0]
        responses[index] = (response, len(response.split()))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(task_wrapper, index, model_name, prompt)
            for index, (model_name, prompt) in enumerate(prompts)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                index = futures.index(future)
                model_name, prompt = prompts[index]
                print(f"Error processing {model_name} with prompt '{prompt}': {e}")

    return responses

def get_multiple_opensource_model_responses(prompts: List[Tuple[str, str]], temperature: float = 0.6, max_tokens: int = 1024) -> List[Tuple[str, int]]:
    """
    Retrieves multiple responses from the open-source models using the specified prompts.

    Args:
        prompts (list): A list of tuples, where each tuple contains the model name and the prompt.
        temperature (float, optional): Controls the randomness of the generated text. Higher values result in more random outputs. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        list: A list of tuples, each containing the generated response text and the number of tokens in the response.
    """
    responses = [None] * len(prompts)
    
    def task_wrapper(index, model_name, prompt):
        response = get_opensource_model_response(model_name, prompt, temperature, max_tokens)[0]
        responses[index] = (response, len(response.split()))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(task_wrapper, index, model_name, prompt)
            for index, (model_name, prompt) in enumerate(prompts)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                index = futures.index(future)
                model_name, prompt = prompts[index]
                print(f"Error processing {model_name} with prompt '{prompt}': {e}")

    return responses

def get_multiple_large_model_responses(prompts: List[Tuple[str, str]], temperature: float = 0.6, max_tokens: int = 1024) -> List[Tuple[str, int]]:
    """
    Retrieves multiple responses from the large models using the specified prompts.

    Args:
        prompts (list): A list of tuples, where each tuple contains the model name and the prompt.
        temperature (float, optional): Controls the randomness of the generated text. Higher values result in more random outputs. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        list: A list of tuples, each containing the generated response text and the number of tokens in the response.
    """
    responses = [None] * len(prompts)
    
    def task_wrapper(index, model_name, prompt):
        response = get_large_model_response(model_name, prompt, temperature, max_tokens)[0]
        responses[index] = (response, len(response.split()))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(task_wrapper, index, model_name, prompt)
            for index, (model_name, prompt) in enumerate(prompts)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                index = futures.index(future)
                model_name, prompt = prompts[index]
                print(f"Error processing {model_name} with prompt '{prompt}': {e}")

    return responses

def get_multiple_other_model_responses(prompts: List[Tuple[str, str]], temperature: float = 0.6, max_tokens: int = 1024) -> List[Tuple[str, int]]:
    """
    Retrieves multiple responses from the other models using the specified prompts.

    Args:
        prompts (list): A list of tuples, where each tuple contains the model name and the prompt.
        temperature (float, optional): Controls the randomness of the generated text. Higher values result in more random outputs. Defaults to 0.6.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1024.

    Returns:
        list: A list of tuples, each containing the generated response text and the number of tokens in the response.
    """
    responses = [None] * len(prompts)
    
    def task_wrapper(index, model_name, prompt):
        response = get_other_model_response(model_name, prompt, temperature, max_tokens)[0]
        responses[index] = (response, len(response.split()))

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(task_wrapper, index, model_name, prompt)
            for index, (model_name, prompt) in enumerate(prompts)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                index = futures.index(future)
                model_name, prompt = prompts[index]
                print(f"Error processing {model_name} with prompt '{prompt}': {e}")

    return responses

model_function = {name: _stub_response for name in model_dict.keys()}

import re

def match_answer(text):
    """
    Find and return all matches of the pattern [[word]] in the given text.
    
    Args:
        text (str): The text to search for matches.
        
    Returns:
        list: A list of matches found in the text.
    """
    pattern = r'\[\[([A-Za-z]+)\]\]'
    matches = re.findall(pattern, text)
    try:
        return matches[-1]
    except:
        return ''

def extract_bracket_content(text):
    pattern = r'\[\[(.*?)\]\]'
    matches = re.findall(pattern, text)
    try:
        return matches[-1]
    except:
        return ''
