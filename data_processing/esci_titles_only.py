import argparse
import os

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ESCI sampled metadata to titles-only JSONL."
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="outputs/esci/sampled_item_metadata_esci.jsonl",
        help="Input JSONL from process_esci.py.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="outputs/esci/esci_titles_only_na2period.jsonl",
        help="Output titles-only JSONL path.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download preprocessed ESCI metadata from HF if input file is missing.",
    )
    parser.add_argument(
        "--fallback-repo-id",
        type=str,
        default="McAuley-Lab/blair-bench",
        help="Fallback repo id used with --download-if-missing.",
    )
    parser.add_argument(
        "--fallback-filename",
        type=str,
        default="processed_esci/sampled_item_metadata_esci.jsonl",
        help="Fallback filename used with --download-if-missing.",
    )
    parser.add_argument(
        "--amazon-meta-repo-id",
        type=str,
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="Repo id with raw_meta_* splits.",
    )
    return parser.parse_args()


def normalize_title(text):
    if pd.isna(text):
        return "."
    text = str(text).strip()
    return text if text else "."


def load_input_jsonl(args):
    path = args.input_jsonl
    if os.path.exists(path):
        return pd.read_json(path, lines=True)

    if not args.download_if_missing:
        raise FileNotFoundError(
            f"Input file not found: {path}. Run process_esci.py first or pass --download-if-missing."
        )

    hf_path = hf_hub_download(
        repo_id=args.fallback_repo_id,
        filename=args.fallback_filename,
        repo_type="dataset",
    )
    return pd.read_json(hf_path, lines=True)


def main():
    args = parse_args()
    df = load_input_jsonl(args)

    item_col = "item_id" if "item_id" in df.columns else "item"
    if item_col not in df.columns:
        raise ValueError("Input JSONL must contain item_id or item column.")
    if "category" not in df.columns:
        raise ValueError("Input JSONL must contain category column.")

    out_df = pd.DataFrame({
        "item_id": df[item_col],
        "category": df["category"],
    })

    out_df["metadata"] = pd.NA

    categories = sorted(out_df["category"].dropna().unique())
    for category in tqdm(categories, desc="Loading category titles"):
        dataset_name = f"raw_meta_{category.replace(' ', '_')}"

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
            mask = out_df["category"] == category
            out_df.loc[mask, "metadata"] = out_df.loc[mask, "item_id"].map(title_map)

        except Exception as e:
            print(f"Failed to process {dataset_name}: {e}")

    out_df["metadata"] = out_df["metadata"].apply(normalize_title)

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    out_df.to_json(args.output_jsonl, orient="records", lines=True, force_ascii=False)

    print(f"Saved ESCI titles-only JSONL to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
