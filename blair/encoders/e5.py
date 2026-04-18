import torch
import torch.nn.functional as F
from vllm import LLM

from blair.encoders.base import BaseSemanticEncoder


class E5Encoder(BaseSemanticEncoder):
    """
    vLLM-based encoder for intfloat/e5-mistral-7b-instruct and Salesforce/SFR-Embedding-Mistral
    """
    def __init__(self, model: str, gpu_id: int, batch_size: int, **kwargs):
        super().__init__(model, gpu_id, batch_size, **kwargs)

        self.model_name = model
        hf_overrides = {
            "head_dim": 128, # 4096 // 32
        }
        self.llm = LLM(
            model=model,
            task="embed",
            trust_remote_code=True,
            enforce_eager=True,
            hf_overrides=hf_overrides
        )
        self.query_task = 'Given a user query describing a need, retrieve relevant entities (such as products, services, or media items) that satisfy the intent and constraints.'

    @property
    def name(self):
        return self.model_name.rsplit("/", 1)[-1]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    @torch.no_grad()
    def encode(self, sentences, query_prompt: bool = False, **kwargs):
        if query_prompt:
            sentences = [self.get_detailed_instruct(self.query_task, s) for s in sentences]
        outputs = self.llm.embed(sentences)
        embeddings = torch.Tensor([o.outputs.embedding for o in outputs])
        return embeddings
