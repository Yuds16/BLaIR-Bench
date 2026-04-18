import torch
import vllm
from vllm import LLM, PoolingParams
from transformers import AutoTokenizer

from blair.encoders.base import BaseSemanticEncoder
from blair.utils import init_device


def truncate_to_token_limit(text: str, tokenizer, max_tokens: int, truncation_strategy: str = "truncate") -> str:
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided to enforce token limits.")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text

    if truncation_strategy == "truncate":
        truncated_ids = token_ids[:max_tokens]
        try:
            return tokenizer.decode(truncated_ids, skip_special_tokens=False)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(truncated_ids)
            if hasattr(tokenizer, "convert_tokens_to_string"):
                return tokenizer.convert_tokens_to_string(tokens)
            return "".join(tokens)

    if truncation_strategy == "raise":
        raise ValueError(
            f"Input text exceeds the maximum of {max_tokens} tokens; received {len(token_ids)} tokens."
        )

    raise ValueError(f"Unsupported truncation_strategy '{truncation_strategy}'.")


class Qwen3Encoder(BaseSemanticEncoder):
    def __init__(self, model, gpu_id, batch_size, dims=-1, **kwargs):
        super().__init__(model, gpu_id, batch_size, **kwargs)

        self.model_name = model
        self.mrl_dim = dims
        if self.mrl_dim > 0:
            self.llm = LLM(model=model, task='embed', hf_overrides={"is_matryoshka": True})
        else:
            self.llm = LLM(model=model, task='embed')
        self.query_task = 'Given a user query describing a need, retrieve relevant entities (such as products, services, or media items) that satisfy the intent and constraints.'
        self.max_tokens = kwargs.get("max_tokens", 32768)
        self.truncation_strategy = kwargs.get("truncation_strategy", "truncate")
        self.tokenizer = self._init_tokenizer()

    @property
    def name(self):
        if self.mrl_dim > 0:
            return f"{self.model_name.rsplit('/', 1)[-1]}-mrl-d{self.mrl_dim}"
        else:
            return self.model_name.rsplit('/', 1)[-1]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def _init_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise tokenizer for {self.model_name}: {exc}") from exc

        if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _prepare_input(self, text: str) -> str:
        return truncate_to_token_limit(text, self.tokenizer, self.max_tokens, self.truncation_strategy)

    @torch.no_grad()
    def encode(self, sentences, query_prompt=False, **kwargs):
        inputs = list(sentences)
        if query_prompt:
            inputs = [self.get_detailed_instruct(self.query_task, s) for s in inputs]

        processed_inputs = [self._prepare_input(text) for text in inputs]

        if self.mrl_dim > 0:
            outputs = self.llm.encode(processed_inputs, pooling_params=PoolingParams(dimensions=self.mrl_dim))
        else:
            outputs = self.llm.encode(processed_inputs)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        return embeddings
