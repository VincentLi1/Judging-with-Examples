import os

from ..base_llm import BaseLLMAPI
from .registry import register_model


@register_model("gemini")
class GEMINI(BaseLLMAPI):
    def __init__(
        self,
        model_pt: str,
        key_path: str | None,
        account_path: str | None,
        parallel_size: int,
        max_retries: int = 10,
        initial_wait_time: int = 2,
        end_wait_time: int = 0,
    ):
        """
        Initializes the BaseLLMAPI object for calling API services.

        Args:
            model_pt (str): Model name
            key_path (str | None): Optional path to a file containing the API key.
            account_path (str | None): Unused placeholder for compatibility with BaseLLMAPI.
            parallel_size (int): Number of parallel processes
            max_retries (int, optional): Maximum number of retries. Defaults to 10.
            initial_wait_time (int, optional): Initial wait time. Defaults to 2.
            end_wait_time (int, optional): End wait time. Defaults to 0.
        """

        from google import genai
        from google.genai import types as genai_types

        self.model_name = model_pt
        api_key = None
        if key_path:
            with open(key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY or pass --api_key_path."
            )

        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time

    def _get_response(
        self,
        prompt: list[dict],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
    ) -> list[dict]:
        """
        Get the response from the API service.

        Args:
            prompt (list[dict]): The prompt to be sent to the API service.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.

        Returns:
            list[dict]: The response from the API service. Each response is a dictionary containing the generated text ("text"), log probabilities ("logprobs", optional), and tokens ("tokens", optional).
        """
        model = self._client.models
        if n > 1:
            print(
                "Warning: GEMINI does not support multiple generations per prompt. Using n=1."
            )
        if logprobs is not None:
            print(
                "Warning: GEMINI does not support log probabilities. Ignoring logprobs."
            )
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        if len([x for x in prompt if x["role"] == "user"]) > 1:
            raise ValueError("Only single round of conversation is supported")
        prompt_text = prompt[-1]["content"]
        config = self._types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            candidate_count=1,
        )
        response = model.generate_content(
            model=self.model_name,
            contents=prompt_text,
            config=config,
        )
        response_text = getattr(response, "text", None)
        if not response_text:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                content = getattr(candidates[0], "content", None)
                if content:
                    parts = getattr(content, "parts", []) or []
                    response_text = "".join(
                        part.text for part in parts if getattr(part, "text", None)
                    )
        response_text = response_text or "No response"
        return [{"text": response_text}]
