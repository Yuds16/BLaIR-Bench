# blair/utils.py

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple
import torch


def init_device(gpu_id, use_torch=True):
    if gpu_id >= 0 and torch.cuda.is_available():
        device_name = f'cuda:{gpu_id}'
    else:
        device_name = 'cpu'
    if use_torch:
        return torch.device(device_name)
    else:
        return device_name


def init_semantic_encoder(model, gpu_id, batch_size, **kwargs):
    def is_bert_like(model):
        return 'bert' in model.lower() or 'roberta' in model.lower()

    if 'sentence-transformers' in model:
        from blair.encoders.sentence_transformer import SentenceTransformerEncoder
        encoder = SentenceTransformerEncoder(model, gpu_id, batch_size, **kwargs)
    elif 'princeton-nlp' in model and 'simcse' in model:
        from blair.encoders.simcse import SimCSEEncoder
        encoder = SimCSEEncoder(model, gpu_id, batch_size, **kwargs)
    elif "gritlm" in model.lower():
        from blair.encoders.gritlm import GritLMEncoder
        encoder = GritLMEncoder(model, gpu_id, batch_size, **kwargs)
    elif is_bert_like(model):
        from blair.encoders.bert_like import BERTLikeEncoder
        encoder = BERTLikeEncoder(model, gpu_id, batch_size, **kwargs)
    elif 'gte' in model:
        from blair.encoders.gte import GTEEncoder
        encoder = GTEEncoder(model, gpu_id, batch_size, **kwargs)
    elif 'text-embedding-3' in model:
        from blair.encoders.text_emb_large_3 import TextEmbEncoder
        if batch_size > 2048 or batch_size is None:
            print("[WARNING] OpenAI API has a max batch size of 2048. Reducing batch size to 2048.")
            batch_size = 2048
        api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key=...")
        encoder = TextEmbEncoder(model=model, batch_size=batch_size, api_key=api_key, **kwargs)
    elif 'gemini-embedding-001' in model:
        from blair.encoders.gemini_emb_001 import GeminiEmbEncoder
        if batch_size > 1024 or batch_size is None:
            print("[WARNING] Gemini API has a max batch size of 1024. Reducing batch size to 1024.")
            batch_size = 1024
        api_key = kwargs.pop("api_key", None) or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key=...")
        encoder = GeminiEmbEncoder(model=model, batch_size=batch_size, api_key=api_key, **kwargs)
    elif 'Qwen3-Embedding' in model:
        from blair.encoders.qwen3 import Qwen3Encoder
        encoder = Qwen3Encoder(model, gpu_id, batch_size, **kwargs)
    elif 'e5' in model or 'SFR-Embedding-Mistral' in model:
        from blair.encoders.e5 import E5Encoder
        encoder = E5Encoder(model, gpu_id, batch_size, **kwargs)
    else:
        raise ValueError(f'Unknown semantic encoder name {model}.')

    print(f'Suffix of the semantic encoder: {encoder.name}')
    return encoder


def apply_pca_if_needed(embeddings: np.ndarray,
                       semantic_encoder_name: str,
                       dataset_name: str,
                       pca_config: dict,
                       cache_dir: str = "./cache/metadata") -> Tuple[np.ndarray, Optional[str]]:
    """
    Convenience function to apply PCA if needed.
    
    Args:
        embeddings: Raw embeddings
        semantic_encoder_name: Name of the semantic encoder (for caching)
        dataset_name: Name of the dataset (for caching)
        pca_config: Dict with keys 'enabled', 'n_components', 'whiten'
        cache_dir: Cache directory for PCA files
        
    Returns:
        Tuple of (processed embeddings, cache path)
    """
    os.makedirs(os.path.join(cache_dir, dataset_name), exist_ok=True)

    n_components = int(pca_config.get('n_components', 0) or 0)
    whiten = bool(pca_config.get('whiten', False))
    force_recompute = bool(pca_config.get('force_recompute', False))
    pca_flag = (
        pca_config.get('enabled', False)
        and 0 < n_components < embeddings.shape[1]
    )

    pca_suffix = f"-pca-d{n_components}-w{whiten}" if pca_flag else ""
    processed_emb_path = os.path.join(
        cache_dir,
        dataset_name,
        f"{dataset_name}.{semantic_encoder_name}{pca_suffix}.npy")

    if not pca_flag:
        # No PCA needed or invalid component count
        return embeddings, processed_emb_path

    if not force_recompute and os.path.exists(processed_emb_path):
        try:
            cached_embeddings = np.load(processed_emb_path)
            if cached_embeddings.shape[0] != embeddings.shape[0]:
                raise ValueError(
                    f"Cached embeddings size mismatch {cached_embeddings.shape} vs {embeddings.shape}"
                )
            print(f"Loading cached PCA embeddings from {processed_emb_path}")
            return cached_embeddings, processed_emb_path
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}")

    print(f"Fitting PCA with n_components={n_components}, whiten={whiten}")
    pca_model = PCA(n_components=n_components, whiten=whiten)
    pca_model.fit(embeddings)

    processed_embeddings = pca_model.transform(embeddings)
    np.save(processed_emb_path, processed_embeddings)
    print(f"Saved PCA-processed embeddings to {processed_emb_path}")

    return processed_embeddings, processed_emb_path
