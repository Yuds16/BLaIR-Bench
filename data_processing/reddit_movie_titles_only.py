import argparse
import ast
import hashlib
import html
import json
import os
import re
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Reddit-Movie query-target-title CSVs as standalone processing."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reddit_movie",
        help="Directory where train/validation/test CSV files will be written.",
    )
    parser.add_argument(
        "--reddit-repo-id",
        type=str,
        default="ZhankuiHe/reddit_movie_large_v1",
        help="Source reddit movie dataset id.",
    )
    parser.add_argument(
        "--id2name-repo-id",
        type=str,
        default="McAuley-Lab/BLaIR-Bench-API",
        help="Repo id for movie id -> movie title mapping.",
    )
    parser.add_argument(
        "--id2name-filename",
        type=str,
        default="prod_search_dls/Reddit_Movie/reddit_movie_id2name.json",
        help="Filename for movie id -> movie title mapping.",
    )
    parser.add_argument("--min-upvotes", type=int, default=20, help="Minimum upvotes filter.")
    return parser.parse_args()


def normalize_conv_group(conv_id):
    if "_" not in conv_id:
        return conv_id
    return conv_id.rsplit("_", 1)[0]


def parse_literal_list(raw_value):
    if isinstance(raw_value, list):
        return raw_value
    if raw_value is None:
        return []
    text = str(raw_value)
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def extract_query(context_raw):
    context = parse_literal_list(context_raw)
    turns = []
    for item in context:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            turns.append(str(item[1]))
        else:
            turns.append(str(item))

    text = " ".join(turns)
    text = html.unescape(text)
    text = text.replace("â\x80\x99", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_conv2turn(dataset):
    conv2turn = defaultdict(set)
    for split in ["train", "validation", "test"]:
        for row in tqdm(dataset[split], desc=f"Scanning seeker turns: {split}"):
            if not row.get("is_seeker", False):
                continue
            conv_group = normalize_conv_group(str(row["conv_id"]))
            conv2turn[conv_group].add(row["turn_id"])
    return conv2turn


def keep_row(row, conv2turn, min_upvotes):
    conv_id = str(row["conv_id"])
    conv_group = normalize_conv_group(conv_id)
    if conv_group not in conv2turn:
        return False

    context_turn_ids = parse_literal_list(row.get("context_turn_ids"))
    if not context_turn_ids:
        return False

    if context_turn_ids[-1] not in conv2turn[conv_group]:
        return False

    if row.get("upvotes", 0) <= min_upvotes:
        return False

    if row.get("processed") == row.get("raw"):
        return False

    return True


def dedup_by_text(rows):
    seen = set()
    deduped = []

    for row in rows:
        text = str(row.get("raw", "")) + str(row.get("context_raw", ""))
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def extract_target_items(processed_text):
    return re.findall(r"tt\d{6,9}", str(processed_text))


def load_id2name(repo_id, filename):
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("id2name mapping file must be a JSON dict.")
    return data


def process_split(dataset_split, conv2turn, id2name, min_upvotes):
    filtered = [
        row
        for row in tqdm(dataset_split, desc="Filtering rows")
        if keep_row(row, conv2turn, min_upvotes)
    ]

    deduped = dedup_by_text(filtered)

    rows = []
    for row in tqdm(deduped, desc="Extracting query-target pairs"):
        query = extract_query(row.get("context_raw"))
        if not query:
            continue

        target_items = sorted(set(extract_target_items(row.get("processed"))))
        for target in target_items:
            if target not in id2name:
                continue
            rows.append(
                {
                    "target": target,
                    "query": query,
                    "metadata": str(id2name[target]).strip(),
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.reddit_repo_id)
    id2name = load_id2name(args.id2name_repo_id, args.id2name_filename)

    conv2turn = build_conv2turn(dataset)

    for split in ["train", "validation", "test"]:
        df = process_split(dataset[split], conv2turn, id2name, args.min_upvotes)
        out_path = os.path.join(args.output_dir, f"{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"{split}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
