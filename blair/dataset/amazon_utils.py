# blair/dataset/utils.py

import os
import re
import glob
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


def _find_local_reviews(domain, cache_dir="cache"):
    """Return the path to a locally cached raw reviews ``<domain>.jsonl`` if any.

    Note the exact-filename match so the metadata file ``meta_<domain>.jsonl`` is
    never picked up here.
    """
    matches = glob.glob(
        os.path.join(cache_dir, "**", f"{domain}.jsonl"), recursive=True
    )
    matches = [m for m in matches if os.path.basename(m) == f"{domain}.jsonl"]
    return matches[0] if matches else None


def load_amazon2023_reviews(domain, max_his_len=None, benchmark="0core_timestamp_w_his"):
    """Build benchmark-style sequential splits from the raw local reviews file.

    Reads the raw ``<domain>.jsonl`` reviews from the cache and reconstructs a
    ``DatasetDict`` with the same ``user_id, parent_asin, history`` columns that
    :func:`load_amazon2023_benchmark` returns, so it is a drop-in replacement for
    the precomputed Hub benchmark split.

    Construction (per user, after de-duplicating ``(user, item)`` pairs and
    sorting chronologically by timestamp) uses a leave-one-out protocol:

      * ``test``  -> last interaction, history = all prior items
      * ``valid`` -> second-to-last interaction, history = all prior items
      * ``train`` -> each earlier interaction, history = its prior items
        (autoregressive expansion, matching the original benchmark)

    ``history`` is the space-joined list of prior ``parent_asin`` values
    (optionally truncated to the most recent ``max_his_len`` items). When no
    local reviews file exists, this falls back to the Hub benchmark split.
    """
    path = _find_local_reviews(domain)
    if path is None:
        print(f"[amazon_utils] No local reviews for '{domain}', using Hub benchmark split...")
        return load_amazon2023_benchmark(domain, benchmark=benchmark)

    print(f"[amazon_utils] Building splits from local reviews: {path}")

    # 1) Collect each user's items with their earliest timestamp (dedup (u, i)).
    user_items = {}  # user_id -> {parent_asin: earliest_timestamp}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dp = json.loads(line)
            except json.JSONDecodeError:
                continue
            u, item, ts = dp.get("user_id"), dp.get("parent_asin"), dp.get("timestamp")
            if not u or not item or ts is None:
                continue
            items = user_items.setdefault(u, {})
            if item not in items or ts < items[item]:
                items[item] = ts

    # 2) Emit leave-one-out rows per user, ordered chronologically.
    splits = {s: {"user_id": [], "parent_asin": [], "history": []}
              for s in ("train", "valid", "test")}

    def _add(split, user, target, history_items):
        if max_his_len is not None:
            history_items = history_items[-max_his_len:]
        splits[split]["user_id"].append(user)
        splits[split]["parent_asin"].append(target)
        splits[split]["history"].append(" ".join(history_items))

    for user, item2ts in user_items.items():
        items = [it for it, _ in sorted(item2ts.items(), key=lambda kv: kv[1])]
        n = len(items)
        if n < 2:
            continue  # need at least one history item + a target
        _add("test", user, items[-1], items[:-1])
        if n >= 3:
            _add("valid", user, items[-2], items[:-2])
        # train targets: items[1 .. n-3] (item[0] would have empty history)
        for k in range(1, n - 2):
            _add("train", user, items[k], items[:k])

    return DatasetDict({
        s: Dataset.from_dict(splits[s]) for s in ("train", "valid", "test")
    })


def _find_local_meta(domain, cache_dir="cache"):
    """Return the path to a locally cached ``meta_<domain>.jsonl`` if one exists.

    Searches the cache tree (e.g. ``cache/cf/<domain>/meta_<domain>.jsonl``) so
    a pre-downloaded metadata file is used instead of hitting the Hub.
    """
    matches = glob.glob(
        os.path.join(cache_dir, "**", f"meta_{domain}.jsonl"), recursive=True
    )
    return matches[0] if matches else None


def _read_meta_jsonl(path):
    """Parse an Amazon Reviews 2023 ``meta_<domain>.jsonl`` file into a Dataset."""
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


def load_amazon2023_meta(domain):
    """Load Amazon Reviews 2023 item metadata.

    Prefers a locally cached ``meta_<domain>.jsonl`` (see :func:`_find_local_meta`)
    so large categories don't have to be re-downloaded. Falls back to fetching
    ``raw/meta_categories/meta_<domain>.jsonl`` from the Hub when no local copy is
    present. Keeps only the flat text fields used for feature construction.
    """
    local_path = _find_local_meta(domain)
    if local_path is not None:
        print(f"[amazon_utils] Loading item metadata from local cache: {local_path}")
        return _read_meta_jsonl(local_path)

    print(f"[amazon_utils] No local metadata for '{domain}', downloading from Hub...")
    path = hf_hub_download(
        _AMAZON_REPO,
        f"raw/meta_categories/meta_{domain}.jsonl",
        repo_type="dataset",
    )
    return _read_meta_jsonl(path)


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
