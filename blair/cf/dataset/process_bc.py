# blair/cf/dataset/process_bc.py

import numpy as np
import pandas as pd
import os
import zipfile
import kagglehub
import shutil
import json

def _download_bc_dataset(raw_data_dir: str) -> None:
    """
    Download Book-Crossing dataset files from Kaggle.
    
    Args:
        raw_data_dir (str): Directory to save the downloaded files
    """
    print(f'[DATASET] Downloading Book-Crossing dataset to {raw_data_dir}')
    os.makedirs(raw_data_dir, exist_ok=True)

    files = [
        "Book reviews/Book reviews/BX-Book-Ratings.csv",
        "Book reviews/Book reviews/BX-Users.csv",
        "Book reviews/Book reviews/BX_Books.csv", # underscore not dash on kaggle
    ]

    # skip if all three already exist
    if all(os.path.exists(os.path.join(raw_data_dir, os.path.basename(f)))
           for f in files):
        print(f"[DATASET] Book-Crossing dataset already exists, skipping download")
        return

    for kaggle_path in files:
        try:
            downloaded_path = kagglehub.dataset_download(
                "ruchi798/bookcrossing-dataset",
                path=kaggle_path,
                force_download=False
            )
            
            # Check if downloaded file is a zip file
            if zipfile.is_zipfile(downloaded_path):
                # extract just the CSV member
                with zipfile.ZipFile(downloaded_path, 'r') as z:
                    # there should be exactly one matching member
                    member = [m for m in z.namelist() if m.endswith(os.path.basename(kaggle_path))][0]
                    z.extract(member, path=raw_data_dir)
                    src = os.path.join(raw_data_dir, member)
                    dst = os.path.join(raw_data_dir, os.path.basename(member))
                    shutil.move(src, dst)
                # clean up the .zip and any empty folders
                os.remove(downloaded_path)
                topdir = os.path.join(raw_data_dir, os.path.dirname(member))
                if os.path.isdir(topdir) and not os.listdir(topdir):
                    os.rmdir(topdir)
                print(f"[DATASET] Pulled & extracted {kaggle_path} → {dst}")
            else:
                # If it's not a zip file, assume it's the CSV file directly
                filename = os.path.basename(kaggle_path)
                dst = os.path.join(raw_data_dir, filename)
                
                # If downloaded_path is a directory, look for the CSV file inside
                if os.path.isdir(downloaded_path):
                    # Find the CSV file in the downloaded directory
                    for root, dirs, files in os.walk(downloaded_path):
                        for file in files:
                            if file.endswith('.csv') and filename.replace('-', '_') in file or filename.replace('_', '-') in file or filename in file:
                                src = os.path.join(root, file)
                                shutil.copy2(src, dst)
                                print(f"[DATASET] Copied {src} → {dst}")
                                break
                    else:
                        # If we can't find the exact file, copy the first CSV that might match
                        csv_files = []
                        for root, dirs, files in os.walk(downloaded_path):
                            csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
                        if csv_files:
                            print(f"[DATASET] Available CSV files: {[os.path.basename(f) for f in csv_files]}")
                            # Try to match based on the filename pattern
                            target_name = os.path.basename(kaggle_path).lower()
                            for csv_file in csv_files:
                                if any(part in os.path.basename(csv_file).lower() for part in target_name.replace('.csv', '').split('-')):
                                    shutil.copy2(csv_file, dst)
                                    print(f"[DATASET] Copied {csv_file} → {dst}")
                                    break
                            else:
                                raise FileNotFoundError(f"Could not find matching CSV file for {kaggle_path}")
                        else:
                            raise FileNotFoundError(f"No CSV files found in {downloaded_path}")
                else:
                    # If it's a file, copy it directly
                    shutil.copy2(downloaded_path, dst)
                    print(f"[DATASET] Copied {downloaded_path} → {dst}")
                    
        except Exception as e:
            print(f"[DATASET] Error downloading {kaggle_path}: {e}")
            raise


def process_bc_cf(
    semantic_encoder,
    output_dir="processed",
):
    """
    1) Download raw Book-Crossing CSVs
    2) Filter & split into train/valid/test
    3) Write .inter files
    4) Build and save data_maps (+ metadata)
    5) Encode item metadata with `semantic_encoder`
    Returns the metadata embedding dimension.
    """

    # reproducibility
    seed = 101   

    cache_dir = os.path.join(output_dir, "Book-Crossing")
    _download_bc_dataset(cache_dir)

    # -- 1) load & rename --
    ratings = pd.read_csv(
        os.path.join(cache_dir, "BX-Book-Ratings.csv"),
        sep=";", encoding="ISO-8859-1",
        escapechar="\\", quotechar='"'
    )
    ratings.rename(
        columns={"User-ID": "user_id", "ISBN": "item_id"},
        inplace=True
    )
    ratings = ratings[["user_id", "item_id"]]

    items = pd.read_csv(
        os.path.join(cache_dir, "BX_Books.csv"),
        sep=";", encoding="ISO-8859-1",
        escapechar="\\", quotechar='"'
    )
    items = items[["ISBN", "Book-Title"]].rename(
        columns={"ISBN": "item_id", "Book-Title": "item_name"}
    )

    # only keep rated items that exist in items list
    ratings = ratings[ratings["item_id"].isin(items["item_id"])]

    # -- 2) filter by activity threshold --
    filter_gate = 20
    # users with ≥ filter_gate interactions
    user_counts = ratings.groupby("user_id").size().reset_index(name="counts")
    good_users = user_counts[user_counts["counts"] >= filter_gate]["user_id"]
    pairs = ratings[ratings["user_id"].isin(good_users)].copy()

    # shuffle & sort
    pairs = pairs.sample(frac=1, random_state=seed).reset_index(drop=True)
    pairs = pairs.sort_values("user_id").reset_index(drop=True)

    # items with ≥ filter_gate interactions
    item_counts = pairs.groupby("item_id").size().reset_index(name="counts")
    good_items = item_counts[item_counts["counts"] >= filter_gate]["item_id"]
    pairs = pairs[pairs["item_id"].isin(good_items)].reset_index(drop=True)

    # -- 3) per-user split 40:30:30 --
    train = pairs.groupby("user_id").sample(frac=0.4, random_state=seed)
    rest  = pairs.drop(train.index)
    valid = rest.groupby("user_id").sample(frac=0.5, random_state=seed)
    test  = rest.drop(valid.index)

    # ensure no new users/items in valid/test
    train_users = set(train["user_id"])
    train_items = set(train["item_id"])
    valid = valid[
        valid["user_id"].isin(train_users) &
        valid["item_id"].isin(train_items)
    ].reset_index(drop=True)
    test = test[
        test["user_id"].isin(train_users) &
        test["item_id"].isin(train_items)
    ].reset_index(drop=True)

    splits = {"train": train, "valid": valid, "test": test}

    # -- 4) write .inter files --
    inter_dir = os.path.join(output_dir, "Book-Crossing")
    os.makedirs(inter_dir, exist_ok=True)
    for name, df_split in splits.items():
        path = os.path.join(inter_dir, f"Book-Crossing.{name}.inter")
        with open(path, "w") as fout:
            fout.write("user_id:token\titem_id:token\n")
            for u, i in zip(df_split["user_id"], df_split["item_id"]):
                fout.write(f"{u}\t{i}\n")

    # -- 5) build ID mappings --
    def remap_id_cf_bc(dfs):
        user2id, id2user = {"[PAD]": 0}, ["[PAD]"]
        item2id, id2item = {"[PAD]": 0}, ["[PAD]"]
        for df_ in dfs:
            for u, i in zip(df_["user_id"], df_["item_id"]):
                if u not in user2id:
                    user2id[u] = len(id2user)
                    id2user.append(u)
                if i not in item2id:
                    item2id[i] = len(id2item)
                    id2item.append(i)
        return {
            "user2id": user2id, "id2user": id2user,
            "item2id": item2id, "id2item": id2item
        }

    data_maps = remap_id_cf_bc(list(splits.values()))

    # attach item metadata
    item2meta = dict(zip(items["item_id"], items["item_name"]))
    id2meta = {0: "[PAD]"}
    for raw_iid, name in item2meta.items():
        mapped = data_maps["item2id"].get(raw_iid)
        if mapped is not None:
            id2meta[mapped] = name
    data_maps["id2meta"] = id2meta

    # save data_maps
    with open(os.path.join(inter_dir, "Book-Crossing.data_maps"), "w") as f:
        json.dump(data_maps, f)

    # -- 6) encode & save metadata embeddings --
    # skip the [PAD] entry at index 0
    sorted_meta = [id2meta[i] for i in range(1, len(data_maps["item2id"]))]
    embeddings = semantic_encoder.encode(sorted_meta)
    meta_path = os.path.join("cache", "metadata", "Book-Crossing", f"Book-Crossing.{semantic_encoder.name}.npy")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    np.save(meta_path, embeddings)
    print(f"Saved new embeddings to: {meta_path}")

    # return embedding dimension
    return embeddings.shape[-1]