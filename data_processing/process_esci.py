import argparse
import html
import json
import os
import random
import re
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone ESCI preprocessing: build test.csv and sampled metadata JSONL."
    )
    parser.add_argument("--n-neg", type=int, default=50, help="Number of negatives per query.")
    parser.add_argument("--n-workers", type=int, default=16, help="Workers used by dataset.map.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/esci",
        help="Output directory for test.csv and sampled_item_metadata_esci.jsonl.",
    )
    parser.add_argument(
        "--asin2cat-repo-id",
        type=str,
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="Repo id containing asin2category.json.",
    )
    parser.add_argument(
        "--asin2cat-filename",
        type=str,
        default="asin2category.json",
        help="asin2category filename.",
    )
    parser.add_argument(
        "--esci-repo-id",
        type=str,
        default="tasksource/esci",
        help="HuggingFace dataset id for ESCI.",
    )
    parser.add_argument(
        "--amazon-meta-repo-id",
        type=str,
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="Repo id with raw_meta_* splits.",
    )
    return parser.parse_args()


def load_asin2category(repo_id, filename):
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                mapping.update(obj)
    return mapping


def clean_text(raw_text):
    if isinstance(raw_text, list):
        text = " ".join(str(x) for x in raw_text)
    elif isinstance(raw_text, dict):
        text = str(raw_text)
    else:
        text = "" if raw_text is None else str(raw_text)

    text = html.unescape(text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r'"', "", text).strip()

    if not text:
        return "."
    return text[:-1] + "." if text.endswith(".") else text + "."


def clean_metadata(example):
    parts = []
    for key in ["title", "description"]:
        if key in example and example[key] is not None:
            parts.append(clean_text(example[key]))

    merged = " ".join(parts).replace("\t", " ").strip()
    example["cleaned_metadata"] = merged if merged else "."
    return example


def main():
    args = parse_args()
    random.seed(args.seed)

    asin2category = load_asin2category(args.asin2cat_repo_id, args.asin2cat_filename)

    category2item = defaultdict(list)
    for asin, cat in asin2category.items():
        if isinstance(cat, str) and "Unknown" not in cat:
            category2item[cat].append(asin)

    os.makedirs(args.output_dir, exist_ok=True)

    query_data = {"qid": [], "query": [], "item_id": []}
    category2items = defaultdict(set)

    raw_dataset = load_dataset(args.esci_repo_id)

    qid = 0
    for row in tqdm(raw_dataset["test"], desc="Filtering ESCI test"):
        item_id = row.get("product_id")

        if item_id not in asin2category:
            continue

        cat = asin2category[item_id]
        if not isinstance(cat, str) or "Unknown" in cat:
            continue

        if row.get("product_locale") != "us":
            continue
        if row.get("esci_label") != "Exact":
            continue
        if row.get("small_version") != 1:
            continue

        category2items[cat].add(item_id)

        neg_pool = category2item[cat]
        if neg_pool:
            k = min(args.n_neg, len(neg_pool))
            for neg_item in random.sample(neg_pool, k):
                category2items[cat].add(neg_item)

        query = str(row.get("query", "")).strip().replace("\t", " ")
        if not query:
            continue

        query_data["qid"].append(qid)
        query_data["query"].append(query)
        query_data["item_id"].append(item_id)
        qid += 1

    query_file = os.path.join(args.output_dir, "test.csv")
    pd.DataFrame(query_data).to_csv(query_file, index=False)

    metadata_file = os.path.join(args.output_dir, "sampled_item_metadata_esci.jsonl")
    with open(metadata_file, "w", encoding="utf-8") as f:
        print(f"Writing metadata to {metadata_file}")
        for category in tqdm(sorted(category2items.keys()), desc="Building metadata JSONL"):
            dataset_name = f"raw_meta_{category.replace(' ', '_')}"

            meta_dataset = load_dataset(
                args.amazon_meta_repo_id,
                dataset_name,
                split="full",
                trust_remote_code=True,
            )
            meta_dataset = meta_dataset.map(clean_metadata, num_proc=args.n_workers)

            candidate_ids = category2items[category]
            for item_id, metadata in zip(meta_dataset["parent_asin"], meta_dataset["cleaned_metadata"]):
                if item_id in candidate_ids:
                    line = {
                        "item_id": item_id,
                        "category": category,
                        "metadata": metadata,
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Saved ESCI query file: {query_file}")
    print(f"Saved ESCI metadata file: {metadata_file}")


if __name__ == "__main__":
    main()
