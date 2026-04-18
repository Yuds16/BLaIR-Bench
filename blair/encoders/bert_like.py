import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from blair.encoders.base import BaseSemanticEncoder
from blair.utils import init_device


class BERTLikeEncoder(BaseSemanticEncoder):
    def __init__(self, model, gpu_id, batch_size, emb_type='CLS', max_length=512, **kwargs):
        super().__init__(model, gpu_id, batch_size, **kwargs)

        self.device = init_device(gpu_id)
        self.emb_type = emb_type
        assert self.emb_type in ["CLS", "Mean"], f"Unknown embedding type: {self.emb_type}"
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.hf_model = AutoModel.from_pretrained(model).to(self.device)
        self.hf_model.eval()

    @property
    def name(self):
        model_name = self.model.rsplit('/', 1)[-1]
        return f'{model_name}-{self.emb_type}'

    def encode(self, sentences, **kwargs):
        all_embeddings = []
        for start_idx in tqdm(range(0, len(sentences), self.batch_size), desc="Encoding"):
            batch_text = sentences[start_idx : start_idx + self.batch_size]
            inputs = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.hf_model(**inputs)

            if self.emb_type == "CLS":
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif self.emb_type == "Mean":
                mask = inputs["attention_mask"].unsqueeze(-1)
                masked_output = outputs.last_hidden_state * mask
                sum_emb = masked_output[:,1:,:].sum(dim=1).cpu().numpy()
                denom = mask[:,1:,:].sum(dim=1).cpu().numpy()
                embeddings = sum_emb / (denom + 1e-8)
            all_embeddings.append(embeddings)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_embeddings
