# blair/prod_search/generate_emb.py

import os
import json
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

def generate_item_emb(
    dataset_name,
    cache_path,
    semantic_encoder,
):
    # 1) Load item metadata
    item_pool = []
    if dataset_name == 'Amazon-C4':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Bench-API',
            filename='prod_search_dls/Amazon-C4/C4_titles_only_na2period.jsonl',
            repo_type='dataset'
        )
    elif dataset_name == 'esci':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Bench-API',
            filename='prod_search_dls/esci/esci_titles_only_na2period.jsonl',
            repo_type='dataset'
        )
    # Load unitest item set
    elif dataset_name == 'testProdSearch':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='test_C4_item.jsonl',
            repo_type='dataset'
        )
    elif dataset_name == 'testApi':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='test_api_item.jsonl',
            repo_type='dataset'
        )
    elif dataset_name == 'reddit_movie':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Bench-API',
            filename='prod_search_dls/Reddit_Movie/reddit_movie_id2name.json',
            repo_type='dataset'
        )
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not supported')

    # Handle different file formats based on dataset
    if dataset_name == 'reddit_movie':
        # For reddit_movie, it's a single JSON dict with movie_id -> title mapping
        with open(filepath, 'r') as file:
            movie_dict = json.load(file)
            item_pool = list(movie_dict.values())  # Use movie titles as metadata
    else:
        # For other datasets, it's JSONL format with metadata field
        with open(filepath, 'r') as file:
            for line in file:
                item_pool.append(json.loads(line.strip())['metadata'])

    # 2) Prepare output path
    out_dir = os.path.join(cache_path, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{dataset_name}.{semantic_encoder.name}")

    if os.path.exists(out_file):
        print(f"[INFO] Embeddings file already exists at {out_file}. Skipping item embeddings generation.")
        features = np.load(out_file)
        print(features.shape)
        emb_size = features.shape[-1]
        return emb_size

    # 3) Encode
    embeddings = semantic_encoder.encode(item_pool)
    emb_size = embeddings.shape[-1]

    # 4) Save
    # Use np.save instead of tofile
    np.save(out_file, embeddings)
    return emb_size


def generate_query_emb(
    dataset_name,
    cache_path,
    semantic_encoder,
):
    
    # Check if we are in a testing environment (set FORCE_DOWNLOAD=1 in your test environment)
    download_mode = "force_redownload" if os.getenv("FORCE_DOWNLOAD", "0") == "1" else None

    # 1) Load queries
    if dataset_name == 'Amazon-C4':
        dataset = load_dataset('McAuley-Lab/Amazon-C4')['test']
        queries = dataset['query']
    elif dataset_name == 'esci':
        filepath = hf_hub_download(
            repo_id="McAuley-Lab/blair-bench",
            filename="processed_esci/test.csv",
            repo_type="dataset",
        )
        dataset = load_dataset("csv", data_files=filepath, download_mode=download_mode)["train"]
        queries = dataset['query']
    # Load unitest query set
    elif dataset_name == 'testProdSearch':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='test_C4_query.jsonl',
            repo_type='dataset'
        )
        dataset = load_dataset("json", data_files=filepath, download_mode=download_mode)["train"]
        queries = dataset['query']
    elif dataset_name == 'testApi':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='test_api_query.jsonl',
            repo_type='dataset'
        )
        dataset = load_dataset("json", data_files=filepath, download_mode=download_mode)["train"]
        queries = dataset['query']
    elif dataset_name == 'reddit_movie':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Bench-API',
            filename='prod_search_dls/Reddit_Movie/test.csv',
            repo_type='dataset'
        )
        dataset = load_dataset("csv", data_files=filepath, download_mode=download_mode)["train"]
        queries = dataset['query'] 
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not supported')

    # 2) Prepare output path
    out_dir = os.path.join(cache_path, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{dataset_name}.q_{semantic_encoder.name}")

    if os.path.exists(out_file):
        print(f"[INFO] Embeddings file already exists at {out_file}. Skipping query embeddings generation.")
        features = np.load(out_file)
        print(features.shape)
        emb_size = features.shape[-1]
        return emb_size

    # 3) Encode
    embeddings = semantic_encoder.encode(queries, query_prompt=True)
    emb_size = embeddings.shape[-1]

    # 4) Save
    # Use np.save instead of tofile
    np.save(out_file, embeddings)
    return emb_size
