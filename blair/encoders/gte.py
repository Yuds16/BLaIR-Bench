import torch
from sentence_transformers import SentenceTransformer

from blair.encoders.base import BaseSemanticEncoder
from blair.utils import init_device


class GTEEncoder(BaseSemanticEncoder):
    def __init__(self, model, gpu_id, batch_size, max_length=8192, **kwargs):
        super().__init__(model, gpu_id, batch_size, **kwargs)

        self.sent_trm = SentenceTransformer(model, device=init_device(gpu_id, use_torch=False), trust_remote_code=True)
        self.sent_trm.max_seq_length = max_length

    @property
    def name(self):
        return self.model.rsplit('/', 1)[-1]

    @torch.no_grad()
    def encode(self, sentences, query_prompt=False, **kwargs):
        encode_kwargs = {
            'batch_size': self.batch_size,
            'show_progress_bar': True,
            **kwargs
        }
        if query_prompt:
            encode_kwargs['prompt_name'] = 'query'
        return self.sent_trm.encode(sentences, **encode_kwargs)
