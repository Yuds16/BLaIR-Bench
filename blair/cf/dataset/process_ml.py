# blair/cf/dataset/process_ml.py

import os
import zipfile
import kagglehub
import shutil
import json

import numpy as np
import pandas as pd

def _download_ml_dataset(raw_data_dir: str) -> None:
    """
    Download the three MovieLens 1M .dat files from Kaggle,
    handling both zipped and raw-file responses.
    """
    print(f'[DATASET] Downloading MovieLens 1M dataset to {raw_data_dir}')
    os.makedirs(raw_data_dir, exist_ok=True)

    files = ["ratings.dat", "users.dat", "movies.dat"]

    # skip if all three already exist
    if all(os.path.exists(os.path.join(raw_data_dir, f)) for f in files):
        print("[DATASET] MovieLens .dat files already present, skipping download")
        return

    for fname in files:
        try:
            downloaded_path = kagglehub.dataset_download(
                "odedgolden/movielens-1m-dataset",
                path=fname,
                force_download=False
            )

            # If we got a ZIP, extract the one member; otherwise it's already the .dat
            if zipfile.is_zipfile(downloaded_path):
                with zipfile.ZipFile(downloaded_path, "r") as z:
                    # find the entry matching our fname
                    member = next(m for m in z.namelist() if m.endswith(fname))
                    z.extract(member, path=raw_data_dir)
                    src = os.path.join(raw_data_dir, member)
                    dst = os.path.join(raw_data_dir, fname)
                    shutil.move(src, dst)
                os.remove(downloaded_path)
                # clean up any empty folder
                topdir = os.path.join(raw_data_dir, os.path.dirname(member))
                if os.path.isdir(topdir) and not os.listdir(topdir):
                    os.rmdir(topdir)
                print(f"[DATASET] Pulled & extracted {fname} → {dst}")
            else:
                # already the raw .dat
                dst = os.path.join(raw_data_dir, fname)
                shutil.move(downloaded_path, dst)
                print(f"[DATASET] Pulled raw file {fname} → {dst}")

        except Exception as e:
            print(f"[DATASET] Error downloading {fname}: {e}")
            raise


def process_ml_cf(
    semantic_encoder,
    output_dir="processed",
):
    """
    1) Download raw MovieLens .dat files
    2) Filter & split into train/valid/test
    3) Write CF-style .inter files
    4) Build & save data_maps (+ movie-title metadata)
    5) Encode item metadata with `semantic_encoder`
    Returns the metadata embedding dimension.
    """

    # reproducibility
    seed = 101   

    cache_dir = os.path.join(output_dir, "ML-1M")
    _download_ml_dataset(cache_dir)

    # -- 1) load & parse .dat files --
    ratings = pd.read_csv(
        os.path.join(cache_dir, "ratings.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )[
        ["user_id", "movie_id"]
    ]

    movies = pd.read_csv(
        os.path.join(cache_dir, "movies.dat"),
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="ISO-8859-1",
    )[
        ["movie_id", "title"]
    ]

    # -- 2) filter by activity threshold --
    filter_gate = 20
    user_counts = ratings.groupby("user_id").size().reset_index(name="counts")
    good_users = set(user_counts[user_counts["counts"] >= filter_gate]["user_id"])
    pairs = ratings[ratings["user_id"].isin(good_users)].copy()

    # shuffle & sort
    pairs = pairs.sample(frac=1, random_state=seed).reset_index(drop=True)
    pairs = pairs.sort_values("user_id").reset_index(drop=True)

    item_counts = pairs.groupby("movie_id").size().reset_index(name="counts")
    good_items = set(item_counts[item_counts["counts"] >= filter_gate]["movie_id"])
    pairs = pairs[pairs["movie_id"].isin(good_items)].reset_index(drop=True)

    # -- 3) per-user split 40:30:30 --
    train = pairs.groupby("user_id").sample(frac=0.4, random_state=seed)
    rest  = pairs.drop(train.index)
    valid = rest.groupby("user_id").sample(frac=0.5, random_state=seed)
    test  = rest.drop(valid.index)

    # ensure no unseen in valid/test
    train_users = set(train["user_id"])
    train_items = set(train["movie_id"])
    valid = valid[
        valid["user_id"].isin(train_users) &
        valid["movie_id"].isin(train_items)
    ].reset_index(drop=True)
    test = test[
        test["user_id"].isin(train_users) &
        test["movie_id"].isin(train_items)
    ].reset_index(drop=True)

    splits = {"train": train, "valid": valid, "test": test}

    # -- 4) write .inter files --
    inter_dir = os.path.join(output_dir, "ML-1M")
    os.makedirs(inter_dir, exist_ok=True)
    for name, df_split in splits.items():
        path = os.path.join(inter_dir, f"ML-1M.{name}.inter")
        with open(path, "w") as fout:
            fout.write("user_id:token\titem_id:token\n")
            for u, i in zip(df_split["user_id"], df_split["movie_id"]):
                fout.write(f"{u}\t{i}\n")

    # -- 5) build ID mappings + attach movie-title metadata --
    def remap_id_cf_ml(dfs):
        u2i, i2u = {"[PAD]": 0}, ["[PAD]"]
        m2i, i2m = {"[PAD]": 0}, ["[PAD]"]
        
        # Collect all unique items first
        all_users = set()
        all_movies = set()
        for df_ in dfs:
            all_users.update(df_["user_id"])
            all_movies.update(df_["movie_id"])
        
        # Sort items for consistent ordering (same as seq_rec would encounter them)
        for movie in sorted(all_movies):
            m2i[movie] = len(i2m)
            i2m.append(movie)
            
        # Process users (order doesn't matter for users since embeddings are item-based)  
        for user in sorted(all_users):
            u2i[user] = len(i2u)
            i2u.append(user)
            
        return {"user2id": u2i, "id2user": i2u, "item2id": m2i, "id2item": i2m}

    data_maps = remap_id_cf_ml(list(splits.values()))

   # metadata: movie titles
    id2meta = {0: "[PAD]"}
    for raw_mid, title in zip(movies["movie_id"], movies["title"]):
        mapped = data_maps["item2id"].get(raw_mid)
        if mapped is not None:
            id2meta[mapped] = title
    data_maps["id2meta"] = id2meta

    # save data_maps
    with open(os.path.join(inter_dir, "ML-1M.data_maps"), "w") as f:
        json.dump(data_maps, f)

    # -- 6) encode & save metadata embeddings (or reuse existing ones) --
    meta_filename = f"ML-1M.{semantic_encoder.name}.npy"
    meta_path = os.path.join("cache", "metadata", "ML-1M", meta_filename)

    # Check if embedding file already exists (e.g., from seq_rec processing)
    if os.path.exists(meta_path):
        print(f"Found existing embedding file: {meta_path}")
        print("Reusing embeddings from previous processing...")
        embeddings = np.load(meta_path)

        # Verify the embedding file has the expected number of items
        expected_items = len(data_maps["item2id"]) - 1  # Exclude [PAD]
        if embeddings.shape[0] != expected_items:
            print(f"WARNING: Embedding file has {embeddings.shape[0]} items, expected {expected_items}")
            print("This might indicate inconsistent item mappings. Regenerating embeddings...")
            embeddings = None
        else:
            print(f"Successfully reused embeddings for {embeddings.shape[0]} items")
    else:
        embeddings = None

    # Generate embeddings if we don't have valid existing ones
    if embeddings is None or embeddings.shape[0] != (len(data_maps["item2id"]) - 1):
        print("Generating new embeddings...")
        sorted_meta = [id2meta[i] for i in range(1, len(data_maps["item2id"]))]
        embeddings = semantic_encoder.encode(sorted_meta)

        # Ensure the metadata directory exists
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        np.save(meta_path, embeddings)
        print(f"Saved new embeddings to: {meta_path}")

    return embeddings.shape[-1]