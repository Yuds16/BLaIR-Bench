import argparse
import json
import os

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Amazon-C4 titles-only metadata as a standalone script."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/amazon_c4_titles_only_na2period.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--c4-repo-id",
        type=str,
        default="McAuley-Lab/Amazon-C4",
        help="HuggingFace repo id for Amazon-C4 sampled metadata.",
    )
    parser.add_argument(
        "--c4-filename",
        type=str,
        default="sampled_item_metadata_1M.jsonl",
        help="Filename in Amazon-C4 repo.",
    )
    parser.add_argument(
        "--asin2cat-repo-id",
        type=str,
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="HuggingFace repo id that contains asin2category.json.",
    )
    parser.add_argument(
        "--asin2cat-filename",
        type=str,
        default="asin2category.json",
        help="asin2category filename.",
    )
    parser.add_argument(
        "--amazon-meta-repo-id",
        type=str,
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="HuggingFace repo id with raw_meta_* datasets.",
    )
    return parser.parse_args()


def load_asin2category(repo_id, filename):
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Some snapshots store this as one JSON dict, others as JSONL.
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


def format_category_name(category):
    return category.replace(" ", "_")


def clean_title(text):
    if pd.isna(text):
        return "."
    text = str(text).strip()
    return text if text else "."


def main():
    args = parse_args()

    c4_path = hf_hub_download(
        repo_id=args.c4_repo_id,
        filename=args.c4_filename,
        repo_type="dataset",
    )
    c4_df = pd.read_json(c4_path, lines=True)

    if "item_id" not in c4_df.columns:
        raise ValueError("Amazon-C4 metadata must contain column 'item_id'.")

    asin2category = load_asin2category(args.asin2cat_repo_id, args.asin2cat_filename)

    df = pd.DataFrame({"item_id": c4_df["item_id"]})
    df["category"] = df["item_id"].map(asin2category)
    df = df.dropna(subset=["category"]).reset_index(drop=True)

    unique_categories = sorted(df["category"].unique())
    category_to_dataset_map = {
        cat: f"raw_meta_{format_category_name(cat)}" for cat in unique_categories
    }

    df["metadata"] = pd.NA

    print(f"Total rows with known categories: {len(df)}")
    print(f"Total categories: {len(unique_categories)}")

    for category, dataset_name in tqdm(
        category_to_dataset_map.items(),
        total=len(category_to_dataset_map),
        desc="Loading category metadata",
    ):
        try:
            dataset = load_dataset(
                args.amazon_meta_repo_id,
                name=dataset_name,
                split="full",
                trust_remote_code=True,
            )

            meta_df = dataset.to_pandas()[["parent_asin", "title"]]
            meta_df = meta_df.rename(columns={"parent_asin": "item_id"})
            meta_df = meta_df.drop_duplicates(subset=["item_id"])

            title_map = pd.Series(meta_df.title.values, index=meta_df.item_id).to_dict()
            category_mask = df["category"] == category
            df.loc[category_mask, "metadata"] = df.loc[category_mask, "item_id"].map(title_map)

        except Exception as e:
            print(f"Failed to process {dataset_name}: {e}")

    df["metadata"] = df["metadata"].apply(clean_title)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_json(args.output, orient="records", lines=True, force_ascii=False)

    print(f"Saved titles-only Amazon-C4 file to: {args.output}")
    print(f"Unique categories in output: {df['category'].nunique()}")


if __name__ == "__main__":
    main()
