# tests/model/test_models.py

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from blair.utils import init_semantic_encoder

# A collection of examples globally including edge case examples.
EXAMPLES = {
    "short": "Love",
    "long": "This is a very long example text intended to test the robustness of the semantic encoder. " * 10,
    "special_chars": "¡Hola! ¿Qué tal? 😊"
}

MODELS = {
    "roberta-large": 1024,
    "bert-base-uncased": 768,
    "sentence-transformers/sentence-t5-large": 768,
    "princeton-nlp/sup-simcse-roberta-large": 1024
}

@pytest.mark.parametrize("model,emb_dim", MODELS.items())
@pytest.mark.parametrize("example_key,text", EXAMPLES.items())
@pytest.mark.parametrize("emb_type", ['Mean', 'CLS'])
def test_model_embeddings(model, emb_dim, example_key, text, emb_type):
    semantic_encoder = init_semantic_encoder(
        model=model,
        gpu_id=0,
        batch_size=32,
        emb_type=emb_type
    )
    
    all_embeddings = semantic_encoder.encode(text)
    assert all_embeddings is not None
    assert all_embeddings.shape[-1] == emb_dim
