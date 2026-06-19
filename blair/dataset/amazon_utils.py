# blair/dataset/utils.py

import os
import re
import html
import json
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import hf_hub_download


# Repo that hosts the Amazon Reviews 2023 data.
_AMAZON_REPO = "McAuley-Lab/Amazon-Reviews-2023"

# Top-level metadata fields that can be turned into text features. Heavy/nested
# fields (images, videos, details, bought_together) are intentionally dropped so
# the resulting table has a flat, Arrow-friendly schema.
_META_TEXT_FIELDS = [
    "parent_asin", "main_category", "title", "average_rating", "rating_number",
    "features", "description", "price", "store", "categories", "subtitle", "author",
]


def load_amazon2023_benchmark(domain, benchmark="0core_timestamp_w_his"):
    """Load Amazon Reviews 2023 benchmark splits directly from the Hub.

    `datasets>=4.0` removed support for dataset loading scripts, so the original
    ``load_dataset("McAuley-Lab/Amazon-Reviews-2023", "0core_timestamp_w_his_<domain>")``
    call no longer works. This reads the same underlying CSV files
    (``benchmark/<kcore>/<split>/<domain>.<split>.csv``) and returns an
    equivalent ``DatasetDict`` with ``train``/``valid``/``test`` splits whose
    columns are ``user_id, parent_asin, rating, timestamp, history`` (all str).
    """
    kcore, split = benchmark.split("_", 1)  # e.g. "0core", "timestamp_w_his"
    splits = {}
    for name in ["train", "valid", "test"]:
        path = hf_hub_download(
            _AMAZON_REPO,
            f"benchmark/{kcore}/{split}/{domain}.{name}.csv",
            repo_type="dataset",
        )
        # dtype=str + keep_default_na=False keeps everything as strings and keeps
        # empty `history` as "" (not NaN), matching the old loading-script output.
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        splits[name] = Dataset.from_pandas(df)
    return DatasetDict(splits)


def load_amazon2023_meta(domain):
    """Load Amazon Reviews 2023 item metadata directly from the Hub.

    Replacement for ``load_dataset(..., "raw_meta_<domain>")`` (see
    :func:`load_amazon2023_benchmark`). Reads
    ``raw/meta_categories/meta_<domain>.jsonl`` and keeps only the flat text
    fields used for feature construction.
    """
    path = hf_hub_download(
        _AMAZON_REPO,
        f"raw/meta_categories/meta_{domain}.jsonl",
        repo_type="dataset",
    )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dp = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = {k: dp.get(k) for k in _META_TEXT_FIELDS}
            # `price` is mixed-typed in the raw data (number, e.g. 28.0, or a
            # string like "from 28.00", or None). Coerce to str so Arrow infers a
            # consistent column type, matching the original loading script.
            if record.get("price") is not None:
                record["price"] = str(record["price"])
            records.append(record)
    return Dataset.from_list(records)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter_items_wo_metadata(example, item2meta):
    if example['parent_asin'] not in item2meta:
        example['history'] = ''
    history = example['history'].split(' ')
    filtered_history = [_ for _ in history if _ in item2meta]
    example['history'] = ' '.join(filtered_history)
    return example


def truncate_history(example, max_his_len):
    example['history'] = ' '.join(example['history'].split(' ')[-max_his_len:])
    return example


def remap_id(datasets):
    user2id = {'[PAD]': 0}
    id2user = ['[PAD]']
    item2id = {'[PAD]': 0}
    id2item = ['[PAD]']

    # Collect all unique items first
    all_items = set()
    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for item_id, history in zip(dataset['parent_asin'], dataset['history']):
            all_items.add(item_id)
            items_in_history = history.split(' ')
            for item in items_in_history:
                if item:  # Skip empty strings
                    all_items.add(item)
    
    # Sort items for consistent ordering (same as cf would encounter them)
    for item in sorted(all_items):
        item2id[item] = len(id2item)
        id2item.append(item)

    # Process users (order doesn't matter for users since embeddings are item-based)
    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for user_id in dataset['user_id']:
            if user_id not in user2id:
                user2id[user_id] = len(id2user)
                id2user.append(user_id)

    data_maps = {'user2id': user2id, 'id2user': id2user, 'item2id': item2id, 'id2item': id2item}
    return data_maps


def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(', '.join(l))
    else:
        return l


def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += '.'
    elif isinstance(feature, list) and len(feature) > 0:
        for v in feature:
            sentence += clean_text(v)
            sentence += ', '
        sentence = sentence[:-2]
        sentence += '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '


def clean_metadata(example, features_needed:list = ['title']):
    meta_text = ''
    for feature in features_needed:
        meta_text += feature_process(example[feature])
    example['cleaned_metadata'] = meta_text
    return example


def process_meta(domain, n_workers, features_needed=['title']):

    meta_dataset = load_amazon2023_meta(domain)

    meta_dataset = meta_dataset.map(
        lambda example: clean_metadata(example, features_needed=features_needed),
        num_proc=n_workers
    )

    item2meta = {}
    for parent_asin, cleaned_metadata in zip(meta_dataset['parent_asin'], meta_dataset['cleaned_metadata']):
        item2meta[parent_asin] = cleaned_metadata

    return item2meta
