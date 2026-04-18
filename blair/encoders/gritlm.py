"""GritLM-7B encoder from https://arxiv.org/abs/2402.09906"""

from blair.encoders.base import BaseSemanticEncoder
from blair.utils import init_device

from gritlm import GritLM


class GritLMEncoder(BaseSemanticEncoder):
    def __init__(
        self,
        model,
        gpu_id,
        batch_size,
        emb_type="CLS",
        max_length=512,
        **kwargs,
    ):
        super().__init__(model, gpu_id, batch_size, **kwargs)

        self.device = init_device(gpu_id)
        self.max_length = max_length
        self.hf_model = GritLM(
            model_name_or_path=model,
            mode="embedding",
            device=self.device,
            normalized=True,
            torch_dtype="auto",
        )
        self.query_instruction = 'Given a user query describing a need, retrieve relevant entities (such as products, services, or media items) that satisfy the intent and constraints.'

    @property
    def name(self):
        return self.model.rsplit('/', 1)[-1]

    def encode(self, sentences, query_prompt=False, **kwargs):
        """Calls `encode` function from https://github.com/ContextualAI/gritlm/blob/df074d65ede1e901b2dd1ab7e9b118de54aee7b6/gritlm/gritlm.py#L93"""
        if query_prompt:
            instruction = self.query_instruction
        else:
            instruction = ""
        return self.hf_model.encode(
            sentences=sentences,
            batch_size=self.batch_size,
            max_length=self.max_length,
            instruction=instruction,
        )
