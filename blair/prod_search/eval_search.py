# blair/prod_search/eval_search.py

import os
# import argparse
import numpy as np
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from recbole.evaluator.metrics import NDCG
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from blair.utils import init_device



def load_items(dataset, data_path):
    """
    Reads item metadata from a .jsonl file (either from Hugging Face Hub or local path),
    populates two mappings:
        id2item (list): index -> item_id
        item2id (dict): item_id -> index
    """
    id2item = []
    item2id = {}

    if dataset == 'Amazon-C4':
        # Download from Hugging Face Hub
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Bench-API',
            filename='prod_search_dls/Amazon-C4/C4_titles_only_na2period.jsonl',
            repo_type='dataset'
        )
    elif dataset == 'esci':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Bench-API',
            filename='prod_search_dls/esci/esci_titles_only_na2period.jsonl',
            repo_type='dataset'
        )
    # Load unitest item set
    elif dataset == 'testProdSearch':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='test_C4_item.jsonl',
            repo_type='dataset'
        )
    elif dataset == 'testApi':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='test_api_item.jsonl',
            repo_type='dataset'
        )
    elif dataset == 'reddit_movie':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Bench-API',
            filename='prod_search_dls/Reddit_Movie/reddit_movie_id2name.json',
            repo_type='dataset'
        )
    else:
        raise NotImplementedError('Dataset not supported: ' + dataset)

    # Handle different file formats based on dataset
    if dataset == 'reddit_movie':
        # For reddit_movie, it's a single JSON dict with movie_id -> title mapping
        with open(filepath, 'r') as file:
            movie_dict = json.load(file)
            for idx, movie_id in enumerate(movie_dict.keys()):
                id2item.append(movie_id)
                item2id[movie_id] = idx
                # Optional sanity-check
                assert len(id2item) == len(item2id)
                assert len(id2item) == idx + 1
    else:
        # For other datasets, it's JSONL format with item_id field
        with open(filepath, 'r') as file:
            for idx, line in enumerate(file):
                item = json.loads(line.strip())['item_id']
                id2item.append(item)
                item2id[item] = idx
                # Optional sanity-check
                assert len(id2item) == len(item2id)
                assert len(id2item) == idx + 1

    return id2item, item2id


def load_queries(dataset, item2id):
    """
    Load the "test" queries from a HF dataset or your local dataset, convert each
    target item_id to an integer index via item2id. Return the list of target indices.
    """
    # Check if we are in a testing environment (set FORCE_DOWNLOAD=1 in your test environment)
    download_mode = "force_redownload" if os.getenv("FORCE_DOWNLOAD", "0") == "1" else None

    query2target = []
    
    if dataset == 'Amazon-C4':
        # Load the official 'test' split from Hugging Face
        dataset_obj = load_dataset('McAuley-Lab/Amazon-C4')['test']
    elif dataset == 'esci':
        # If we have a test.csv in the HF Hub
        filepath = hf_hub_download(
            repo_id="McAuley-Lab/blair-bench",
            filename="processed_esci/test.csv",
            repo_type="dataset",
        )
        dataset_obj = load_dataset("csv", data_files=filepath, download_mode=download_mode)["train"]
    elif dataset == 'testProdSearch':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='test_C4_query.jsonl',
            repo_type='dataset'
        )
        dataset_obj = load_dataset("json", data_files=filepath, download_mode=download_mode)["train"]
    elif dataset == 'testApi':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='test_api_query.jsonl',
            repo_type='dataset'
        )
        dataset_obj = load_dataset("json", data_files=filepath, download_mode=download_mode)["train"]
    elif dataset == 'reddit_movie':
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Bench-API',
            filename='prod_search_dls/Reddit_Movie/test.csv',
            repo_type='dataset'
        )
        dataset_obj = load_dataset("csv", data_files=filepath, download_mode=download_mode)["train"]
    else:
        raise NotImplementedError('Dataset not supported: ' + dataset)
    
    # For each row, map the item_id to an item index
    # Handle different column names based on dataset
    if dataset == 'reddit_movie':
        target_column = 'target'  # reddit_movie uses 'target' column
    else:
        target_column = 'item_id'  # other datasets use 'item_id' column

    for target_item in dataset_obj[target_column]:
        target_id = item2id[target_item]
        query2target.append(target_id)

    return query2target


def load_plm_embedding(data_path, dataset_name, suffix, plm_size):
    """
    Load embeddings from a flat binary file:
      {dataset_name}.{suffix}
    Reshape them into (num_items, plm_size).
    """
    feat_path = os.path.join(data_path, dataset_name, f'{dataset_name}.{suffix}')
    loaded_feat = np.load(feat_path).reshape(-1, plm_size)
    return torch.FloatTensor(loaded_feat)


def evaluate_search(
    dataset='Amazon-C4',
    suffix='blair-baseCLS',
    k=100,
    gpu_id=0,
    batch_size=64,
    domain=False,
    data_path="./cache",
    emb_size = 768
):
    """
    A refactored version of the original eval_search.py that doesn't use argparse,
    but instead takes direct function arguments and runs the retrieval evaluation.
    """

    # 1) Prepare
    device = init_device(gpu_id)
    dataset_name = dataset.split('/')[-1]  # e.g., 'Amazon-C4' or 'esci'
    plm_size = emb_size

    # 2) Load items and queries
    id2item, item2id = load_items(dataset, data_path)
    query2target_list = load_queries(dataset, item2id)
    query2target = torch.LongTensor(query2target_list).to(device)

    # 3) Load item and query embeddings
    #    e.g. "Amazon-C4.blair-baseCLS" for items, "Amazon-C4.q_blair-baseCLS" for queries
    item_embs = load_plm_embedding(data_path, dataset_name, suffix, plm_size)
    item_embs = F.normalize(item_embs, dim=-1).to(device)

    query_embs = load_plm_embedding(data_path, dataset_name, 'q_' + suffix, plm_size)
    query_embs = F.normalize(query_embs, dim=-1).to(device)

    # 4) Check shape
    assert item_embs.shape[0] == len(id2item), "Mismatch in item embeddings and metadata!"
    assert query_embs.shape[0] == len(query2target), "Mismatch in query embeddings and queries!"

    # 5) Set up NDCG@k metric
    metric = NDCG({
        'metric_decimal_place': 4,
        'topk': k
    })

    # 6) Inference loop - compute similarity, retrieve top-k, compute NDCG
    results = []
    with torch.no_grad():
        for pr in tqdm(range(0, query_embs.shape[0], batch_size), desc="Evaluating"):
            batch_queries = query_embs[pr:pr+batch_size]
            batch_target = query2target[pr:pr+batch_size]

            # Scores: [batch_size, n_items]
            scores = batch_queries @ item_embs.T

            topk_scores, topk_indices = torch.topk(scores, k, dim=-1)  # top-k items
            pos_index = (batch_target.unsqueeze(-1).expand(-1, k) == topk_indices).cpu()
            pos_len = torch.ones_like(batch_target).cpu().numpy()

            ndcg = metric.metric_info(pos_index, pos_len)
            # ndcg has shape [batch_size, k], we want final column -> NDCG@k
            ndcg_at_k = ndcg[:, -1]
            results.append(ndcg_at_k)

    # 7) Combine results and print
    results = np.concatenate(results)
    overall_ndcg = results.mean()
    print(f"Dataset = {dataset}, Suffix = {suffix}, K = {k}, Batch Size = {batch_size}")
    print(f"Overall NDCG@{k}: {overall_ndcg:.4f}")
        
    final_results = {
        "Dataset": dataset,
        "Suffix": suffix,
        "K": k,
        "Batch Size": batch_size,
        f"Overall NDCG@{k}": f"{overall_ndcg:.4f}"
    }
    return final_results
