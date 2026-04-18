import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from blair.encoders.base import BaseSemanticEncoder
from blair.utils import init_device

class SimCSEEncoder(BaseSemanticEncoder):
    def __init__(self, model_name, gpu_id, batch_size, **kwargs):
        super().__init__(model_name, gpu_id, batch_size, **kwargs)
        # device string like "cuda:0" or "cpu"
        self.device = init_device(gpu_id, use_torch=True)
        
        # load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.simcse = AutoModel.from_pretrained(model_name).to(self.device)
        self.simcse.eval()

    @property
    def name(self):
        # just the last component of the model name
        return self.model.rsplit('/', 1)[-1]

    def encode(self, sentences, **kwargs):
        """
        sentences: List[str]
        returns: numpy array of shape (len(sentences), hidden_size)
        """
        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), self.batch_size), desc=f"SimCSE[{self.name}]"):
                batch_sents = sentences[i:i + self.batch_size]
                inputs = self.tokenizer(
                    batch_sents,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                # forward pass: we only need last_hidden_state
                outputs = self.simcse(**inputs, return_dict=True)
                # SimCSE uses the [CLS] token output
                cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, D)
                # Normalize each row to unit length since pretrained with cosine loss
                cls_emb = F.normalize(cls_emb, p=2, dim=1)
                all_embeds.append(cls_emb.cpu().numpy())

        return np.vstack(all_embeds)
