# blair/seq_rec/dataset/process_amazon_2023.py

import os
import re
import html
import json
import numpy as np
from datasets import load_dataset

from blair.dataset.amazon_utils import check_path, filter_items_wo_metadata, truncate_history, remap_id, process_meta, load_amazon2023_reviews


def process_amazon(
    domain="All_Beauty",
    max_his_len=50,
    n_workers=16,
    output_dir="processed",
    device="cuda:0",
    semantic_encoder="hyp1231/blair-roberta-base",
    batch_size=16,
    features_needed=['title'],
):
    """
    A Python function that loads & processes data for a given domain and semantic encoder.
    Replaces the old CLI approach from if __name__ == '__main__'.
    """

    # 1) Load main dataset (build sequential splits from the raw local reviews)
    datasets = load_amazon2023_reviews(domain, max_his_len=max_his_len)

    # 2) Process meta
    item2meta = process_meta(domain, n_workers, features_needed)

    truncated_datasets = {}
    domain_output_dir = os.path.join(output_dir, domain)
    check_path(domain_output_dir)

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

        output_path = os.path.join(domain_output_dir, f'{domain}.{split}.inter')
        with open(output_path, 'w') as f:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for user_id, history, parent_asin in zip(
                truncated_dataset['user_id'],
                truncated_dataset['history'],
                truncated_dataset['parent_asin']
            ):
                f.write(f"{user_id}\t{history}\t{parent_asin}\n")

    # Remap IDs
    data_maps = remap_id(truncated_datasets)
    id2meta = {0: '[PAD]'}
    for item in item2meta:
        if item not in data_maps['item2id']:
            continue
        item_id = data_maps['item2id'][item]
        id2meta[item_id] = item2meta[item]
    data_maps['id2meta'] = id2meta

    # Save data_maps
    data_maps_path = os.path.join(domain_output_dir, f'{domain}.data_maps')
    with open(data_maps_path, 'w') as f:
        json.dump(data_maps, f)

    # Encode metadata into embeddings (or reuse existing ones)
    feat_file = f'{domain}.{semantic_encoder.name}'
    emb_file_path = os.path.join("cache", "metadata", domain, feat_file + '.npy')
    
    # Check if embedding file already exists (e.g., from cf processing)
    if os.path.exists(emb_file_path):
        print(f"Found existing embedding file: {emb_file_path}")
        print("Reusing embeddings from previous processing...")
        all_embeddings = np.load(emb_file_path)
        emb_size = all_embeddings.shape[-1]
        
        # Verify the embedding file has the expected number of items
        expected_items = len(data_maps['item2id']) - 1  # Exclude [PAD]
        if all_embeddings.shape[0] != expected_items:
            print(f"WARNING: Embedding file has {all_embeddings.shape[0]} items, expected {expected_items}")
            print("This might indicate inconsistent item mappings. Regenerating embeddings...")
            all_embeddings = None
        else:
            print(f"Successfully reused embeddings for {all_embeddings.shape[0]} items")
    else:
        all_embeddings = None
    
    # Generate embeddings if we don't have valid existing ones
    if all_embeddings is None or all_embeddings.shape[0] != (len(data_maps['item2id']) - 1):
        print("Generating new embeddings...")
        # 1) Build a sorted list of metadata text for items, skipping item_id=0 => [PAD]
        sorted_text = []
        for i in range(1, len(data_maps['item2id'])):
            sorted_text.append(data_maps['id2meta'][i])

        # 2) Encode the item metadata (always save raw embeddings)
        all_embeddings = semantic_encoder.encode(sorted_text)
        emb_size = all_embeddings.shape[-1]

        # Use np.save instead of tofile
        metadata_dir = os.path.join("cache", "metadata", domain)
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, feat_file + '.npy')
        np.save(metadata_path, all_embeddings)
        print(f"Saved new embeddings to: {metadata_path}")

    # Some basic stats
    print(f"#Users: {len(data_maps['user2id']) - 1}")
    print(f"#Items: {len(data_maps['item2id']) - 1}")

    n_interactions = {}
    for split in ['train', 'valid', 'test']:
        n_interactions[split] = len(truncated_datasets[split])
        for history in truncated_datasets[split]['history']:
            if len(history.split(' ')) == 1:
                n_interactions[split] += 1
    print(f"#Interaction in total: {sum(n_interactions.values())}")
    print(n_interactions)

    avg_his_length = 0
    for split in ['train', 'valid', 'test']:
        avg_his_length += sum([len(_.split(' ')) for _ in truncated_datasets[split]['history']])
    avg_his_length /= sum([len(truncated_datasets[split]) for split in ['train', 'valid', 'test']])
    print(f"Average history length: {avg_his_length}")
    # print(f"Average character length of metadata: {np.mean([len(_) for _ in sorted_text])}")

    # Return anything you want. For instance, the path of the feature file:
    return emb_size # os.path.join(domain_output_dir, feat_file)
