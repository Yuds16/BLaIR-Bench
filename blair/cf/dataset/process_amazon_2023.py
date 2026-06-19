# blair/cf/dataset/process_amazon_2023.py

import os
import re
import html
import json
import numpy as np
import pandas as pd
from datasets import load_dataset

from blair.dataset.amazon_utils import check_path, filter_items_wo_metadata, truncate_history, remap_id, process_meta, load_amazon2023_benchmark


def process_amazon_cf(
    domain="All_Beauty",
    max_his_len=50,
    n_workers=16,
    output_dir="processed",
    device="cuda:0",
    semantic_encoder="hyp1231/blair-roberta-base",
    batch_size=16,
    features_needed=['title'],
):
    # 1) Load main dataset
    datasets = load_amazon2023_benchmark(domain, benchmark="0core_timestamp_w_his")

    # 2) Process meta
    item2meta = process_meta(domain, n_workers, features_needed)

    truncated_datasets = {}
    domain_output_dir = os.path.join(output_dir, domain)
    check_path(domain_output_dir)

    # Collect all unique (user_id, item_id) pairs across all original splits
    pairs_set = set()
    interaction_count = {split: 0 for split in ['train', 'valid', 'test']}

    for split in ['train', 'valid', 'test']:
        # Remove lines w/ empty history
        filtered_dataset = datasets[split].map(
            lambda t: filter_items_wo_metadata(t, item2meta),
            num_proc=n_workers
        )
        filtered_dataset = filtered_dataset.filter(lambda t: len(t['history']) > 0)
        # Truncate history
        truncated_dataset = filtered_dataset.map(
            lambda t: truncate_history(t, max_his_len),
            num_proc=n_workers
        )
        truncated_datasets[split] = truncated_dataset

        # Accumulate unique pairs from both history and the current target item
        for user_id, history, parent_asin in zip(
            truncated_datasets[split]['user_id'],
            truncated_datasets[split]['history'],
            truncated_datasets[split]['parent_asin']
        ):
            # expand history into pairs
            for item in history.split(' '):
                if not item:
                    continue
                pairs_set.add((user_id, item))
            # include the current item as an interaction
            pairs_set.add((user_id, parent_asin))

    # After collecting all pairs, randomly split into 4:3:3 (train:valid:test)
    all_pairs = list(pairs_set)
    rng = np.random.default_rng(101)  # fixed seed for reproducibility
    rng.shuffle(all_pairs)

    n_total = len(all_pairs)
    n_train = int(n_total * 0.4)
    n_valid = int(n_total * 0.3)
    n_test = n_total - n_train - n_valid

    split_pairs = {
        'train': all_pairs[:n_train],
        'valid': all_pairs[n_train:n_train + n_valid],
        'test': all_pairs[n_train + n_valid:]
    }

    # Write the new splits to .inter files
    for split in ['train', 'valid', 'test']:
        output_path = os.path.join(domain_output_dir, f'{domain}.{split}.inter')
        with open(output_path, 'w') as f:
            f.write('user_id:token\titem_id:token\n')
            for u, i in split_pairs[split]:
                f.write(f"{u}\t{i}\n")
        interaction_count[split] = len(split_pairs[split])

    # 4) Build ID mappings over the CF splits
    data_maps = remap_id(truncated_datasets)
    id2meta = {0: '[PAD]'}
    for item, text in item2meta.items():
        if item in data_maps["item2id"]:
            id2meta[data_maps["item2id"][item]] = text
    data_maps["id2meta"] = id2meta

    # 5) Save data_maps
    with open(os.path.join(domain_output_dir, f"{domain}.data_maps"), "w") as f:
        json.dump(data_maps, f)

    # 6) Encode metadata into embeddings (or reuse existing ones)
    feat_file = f"{domain}.{semantic_encoder.name}"
    emb_file_path = os.path.join("cache", "metadata", domain, feat_file + '.npy')
    
    # Check if embedding file already exists (e.g., from seq_rec processing)
    if os.path.exists(emb_file_path):
        print(f"Found existing embedding file: {emb_file_path}")
        print("Reusing embeddings from previous processing...")
        all_embeddings = np.load(emb_file_path)
        emb_size = all_embeddings.shape[-1]
        
        # Verify the embedding file has the expected number of items
        expected_items = len(data_maps["item2id"]) - 1  # Exclude [PAD]
        if all_embeddings.shape[0] != expected_items:
            print(f"WARNING: Embedding file has {all_embeddings.shape[0]} items, expected {expected_items}")
            print("This might indicate inconsistent item mappings. Regenerating embeddings...")
            # Fall through to regenerate embeddings
        else:
            print(f"Successfully reused embeddings for {all_embeddings.shape[0]} items")
    else:
        all_embeddings = None
    
    # Generate embeddings if we don't have valid existing ones
    if all_embeddings is None or all_embeddings.shape[0] != (len(data_maps["item2id"]) - 1):
        print("Generating new embeddings...")
        sorted_text = [data_maps["id2meta"][i]
                       for i in range(1, len(data_maps["item2id"]))]
        all_embeddings = semantic_encoder.encode(sorted_text)
        emb_size = all_embeddings.shape[-1]
        metadata_dir = os.path.join("cache", "metadata", domain)
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, feat_file + '.npy')
        np.save(metadata_path, all_embeddings)
        print(f"Saved new embeddings to: {metadata_path}")

    # 8) Print stats
    print(f"#Users: {len(data_maps['user2id']) - 1}")
    print(f"#Items: {len(data_maps['item2id']) - 1}")

    print(f"#Interaction in total: {sum(interaction_count.values())}")
    print(f"Interactions by split (random 4:3:3): {interaction_count}")

    return emb_size