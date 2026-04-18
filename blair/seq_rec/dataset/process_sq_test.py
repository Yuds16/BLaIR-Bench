# blair/seq_rec/dataset/process_sq_test.py

import os
import re
import html
import json
import numpy as np

from datasets import load_dataset, DatasetDict
from blair.dataset.amazon_utils import check_path, filter_items_wo_metadata, truncate_history, remap_id, list_to_str, clean_text, feature_process, clean_metadata, process_meta
from huggingface_hub import hf_hub_download

def process_sq_test(
    max_his_len=50,
    n_workers=16,
    output_dir="processed",
    device="cuda:0",
    semantic_encoder="hyp1231/blair-roberta-base",
    batch_size=16,
):
    """
    A Python function that loads & processes data for a given domain and semantic encoder.
    Replaces the old CLI approach from if __name__ == '__main__'.
    """

    # Check if we are in a testing environment (set FORCE_DOWNLOAD=1 in your test environment)
    download_mode = "force_redownload" if os.getenv("FORCE_DOWNLOAD", "0") == "1" else None

    filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='seq_rec_train_top10.jsonl',
            repo_type='dataset'
        )
    dataset_train = load_dataset("json", data_files=filepath, download_mode=download_mode)["train"]

    filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='seq_rec_valid_top10.jsonl',
            repo_type='dataset'
        )
    dataset_valid = load_dataset("json", data_files=filepath, download_mode=download_mode)["train"]

    filepath = hf_hub_download(
            repo_id='McAuley-Lab/BLaIR-Benchmark-Testset',
            filename='seq_rec_test_top10.jsonl',
            repo_type='dataset'
        )
    dataset_test = load_dataset("json", data_files=filepath, download_mode=download_mode)["train"]

    datasets = DatasetDict({
        "train": dataset_train,
        "valid": dataset_valid,
        "test": dataset_test
    })

    # 2) Process meta
    item2meta = process_meta("All_Beauty", n_workers)  # or pass domain, n_workers, etc.

    truncated_datasets = {}
    domain_output_dir = os.path.join(output_dir, "testSeqRec")
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

        # Save the truncated dataset into the dictionary
        truncated_datasets[split] = truncated_dataset

        output_path = os.path.join(domain_output_dir, f'testSeqRec.{split}.inter')
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
    data_maps_path = os.path.join(domain_output_dir, 'testSeqRec.data_maps')
    with open(data_maps_path, 'w') as f:
        json.dump(data_maps, f)

    # 1) Build a sorted list of metadata text for items, skipping item_id=0 => [PAD]
    sorted_text = []
    for i in range(1, len(data_maps['item2id'])):
        sorted_text.append(data_maps['id2meta'][i])

    # 2) Encode the item metadata
    all_embeddings = semantic_encoder.encode(sorted_text)
    emb_size = all_embeddings.shape[-1]

    # Use np.save instead of tofile
    feat_file = f'testSeqRec.{semantic_encoder.name}'
    np.save(os.path.join(domain_output_dir, feat_file), all_embeddings)

    # Some basic stats
    print(f"#Users: {len(data_maps['user2id']) - 1}")
    print(f"#Items: {len(data_maps['item2id']) - 1}")

    n_interactions = {}
    for split in ['train', 'valid', 'test']:
        n_interactions[split] = len(truncated_datasets[split])
        for history in truncated_datasets[split]['history']:
            # This might be a mistake testing == 1 instead of >= 1
            if len(history.split(' ')) == 1:
                n_interactions[split] += 1
    print(f"#Interaction in total: {sum(n_interactions.values())}")
    print(n_interactions)

    avg_his_length = 0
    for split in ['train', 'valid', 'test']:
        avg_his_length += sum([len(_.split(' ')) for _ in truncated_datasets[split]['history']])
    avg_his_length /= sum([len(truncated_datasets[split]) for split in ['train', 'valid', 'test']])
    print(f"Average history length: {avg_his_length}")
    print(f"Average character length of metadata: {np.mean([len(_) for _ in sorted_text])}")

    # Return anything you want. For instance, the path of the feature file:
    return emb_size # os.path.join(domain_output_dir, feat_file)