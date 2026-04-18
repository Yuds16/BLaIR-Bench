import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from blair.encoders.base import BaseSemanticEncoder
from blair.utils import init_device


class SentenceTransformerEncoder(BaseSemanticEncoder):
    def __init__(self, model, gpu_id, batch_size, **kwargs):
        super().__init__(model, gpu_id, batch_size, **kwargs)

        self.sent_trm = SentenceTransformer(model, device=init_device(gpu_id, use_torch=False))

    @property
    def name(self):
        return self.model.rsplit('/', 1)[-1]

    def encode(self, sentences, **kwargs):
        # Additional pooling is not needed for sent_trm models
        return self.sent_trm.encode(
            sentences, 
            batch_size=self.batch_size,
            show_progress_bar=True
        )
