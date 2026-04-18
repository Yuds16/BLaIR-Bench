# Adding a New Encoder to BLaIR-Bench

Thanks for contributing to the BLaIR-Bench project!

BLaIR-Bench is a toolkit for evaluating semantic encoders (PLMs) in encoding item text features for recommendation and product search. Below is a step-by-step guide for adding a new encoder.

## 1. Create a New Encoder Class

Create a new file under `blair/encoders/`, e.g., `blair/encoders/my_encoder.py`.

Your class must inherit from `BaseSemanticEncoder` and implement two things:

### The `name` Property

Returns a string identifier for this encoder. Include any important configuration in the name (e.g., pooling strategy), but omit defaults that rarely change.

```python
@property
def name(self):
    return f'{self.model}-{self.emb_type}'
```

### The `encode` Method

Takes a list of N sentences and returns a NumPy array of shape `(N, embedding_dim)`.

### Full Example

Here is a complete example based on `blair/encoders/bert_like.py`:

```python
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
        return f'{self.model}-{self.emb_type}'

    def encode(self, sentences):
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
```

Custom arguments (e.g., `emb_type`, `max_length`) should have default values. Most HuggingFace model pages include example encoding code you can adapt.

## 2. Register in the Routing Function

Update the factory function in `blair/utils.py` so that it dispatches to your new class based on the model name:

```python
def init_semantic_encoder(model, gpu_id, batch_size, **kwargs):
    # ... existing conditions ...
    elif is_my_model(model):
        from blair.encoders.my_encoder import MyEncoder
        encoder = MyEncoder(model, gpu_id, batch_size, **kwargs)
    # ...
```

## 3. Update the README

Add your encoder to the supported models list in `README.md`.

## 4. Submit a Pull Request

1. Create a new branch
2. Commit your changes
3. Push and open a PR to merge into `main`

If you run into any issues, feel free to reach out to [yphou@ucsd.edu](mailto:yphou@ucsd.edu).
