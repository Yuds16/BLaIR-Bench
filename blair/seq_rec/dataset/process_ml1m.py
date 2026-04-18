# blair/seq_rec/dataset/process_ml1m.py

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


def process_ml_sq(
    semantic_encoder,
    output_dir="processed",
    valid_size: int = 1,
    test_size: int  = 1,
    max_seq_len: int = 200,
):
    """
    1) Download raw MovieLens .dat files
    2) Filter by activity threshold
    3) Build time-ordered per-user sequences
    4) Write sequential train/valid/test .inter files
    5) Build & save data_maps (+ movie-title metadata)
    6) Encode item metadata with `semantic_encoder`
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
        ["user_id", "movie_id", "timestamp"]
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
    movie_titles = dict(zip(movies["movie_id"], movies["title"]))

    # -- 2) filter by activity threshold --
    filter_gate = 20
    user_counts = ratings.groupby("user_id").size().reset_index(name="counts")
    good_users = set(user_counts[user_counts["counts"] >= filter_gate]["user_id"])
    pairs = ratings[ratings["user_id"].isin(good_users)].copy()

    item_counts = pairs.groupby("movie_id").size().reset_index(name="counts")
    good_items = set(item_counts[item_counts["counts"] >= filter_gate]["movie_id"])
    pairs = pairs[pairs["movie_id"].isin(good_items)].reset_index(drop=True)

    # -- 3) build time-ordered sequences --
    pairs = pairs.sort_values(["user_id", "timestamp"])
    user_seqs = pairs.groupby("user_id")["movie_id"].apply(list).to_dict()

    # TRAIN: all prefix→next from start .. -(valid+test)
    with open(os.path.join(cache_dir, "ML-1M.train.inter"), "w") as fout:
        fout.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for u, seq in user_seqs.items():
            if len(seq) < valid_size + test_size + 1:
                continue
            train_seq = seq[:-(valid_size + test_size)]
            for i in range(1, len(train_seq)):
                prefix = train_seq[max(0, i - max_seq_len):i]
                target = train_seq[i]
                fout.write(f"{u}\t{' '.join(map(str, prefix))}\t{target}\n")

    # VALID: full train history → predict the held‐out valid item
    with open(os.path.join(cache_dir, "ML-1M.valid.inter"), "w") as fout:
        fout.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for u, seq in user_seqs.items():
            if len(seq) < valid_size + test_size + 1:
                continue
            hist = seq[:-(valid_size + test_size)]
            hist = hist[-max_seq_len:]
            val_item = seq[-(valid_size + test_size)]
            fout.write(f"{u}\t{' '.join(map(str, hist))}\t{val_item}\n")

    # TEST: train+valid history → predict the held‐out test item
    with open(os.path.join(cache_dir, "ML-1M.test.inter"), "w") as fout:
        fout.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for u, seq in user_seqs.items():
            if len(seq) < valid_size + test_size + 1:
                continue
            hist = seq[:-test_size]
            hist = hist[-max_seq_len:]
            test_item = seq[-1]
            fout.write(f"{u}\t{' '.join(map(str, hist))}\t{test_item}\n")

    # -- 5) build ID mappings + attach movie-title metadata --
    user2id, id2user = {"[PAD]": 0}, ["[PAD]"]
    item2id, id2item = {"[PAD]": 0}, ["[PAD]"]

    # Collect all unique items first
    all_users = set()
    all_movies = set()
    for u, seq in user_seqs.items():
        all_users.add(u)
        all_movies.update(seq)
        
    # Sort items for consistent ordering (same as cf would encounter them)
    for movie in sorted(all_movies):
        item2id[movie] = len(id2item)
        id2item.append(movie)
        
    # Process users (order doesn't matter for users since embeddings are item-based)
    for user in sorted(all_users):
        user2id[user] = len(id2user)
        id2user.append(user)

    data_maps = {
        "user2id": user2id,
        "id2user": id2user,
        "item2id": item2id,
        "id2item": id2item,
    }

    # metadata: movie titles
    id2meta = {0: "[PAD]"}
    for raw_mid, idx in item2id.items():
        if raw_mid == "[PAD]":
            continue
        id2meta[idx] = movie_titles.get(raw_mid, "")

    data_maps["id2meta"] = id2meta

    # save data_maps
    with open(os.path.join(cache_dir, "ML-1M.data_maps"), "w") as f:
        json.dump(data_maps, f)

    # -- 6) encode & save metadata embeddings (or reuse existing ones) --
    meta_filename = f"ML-1M.{semantic_encoder.name}.npy"
    meta_path = os.path.join("cache", "metadata", "ML-1M", meta_filename)
    
    # Check if embedding file already exists (e.g., from cf processing)
    if os.path.exists(meta_path):
        print(f"Found existing embedding file: {meta_path}")
        print("Reusing embeddings from previous processing...")
        embeddings = np.load(meta_path)
        
        # Verify the embedding file has the expected number of items
        expected_items = len(id2meta) - 1  # Exclude [PAD]
        if embeddings.shape[0] != expected_items:
            print(f"WARNING: Embedding file has {embeddings.shape[0]} items, expected {expected_items}")
            print("This might indicate inconsistent item mappings. Regenerating embeddings...")
            embeddings = None
        else:
            print(f"Successfully reused embeddings for {embeddings.shape[0]} items")
    else:
        embeddings = None
    
    # Generate embeddings if we don't have valid existing ones
    if embeddings is None or embeddings.shape[0] != (len(id2meta) - 1):
        print("Generating new embeddings...")
        sorted_meta = [id2meta[i] for i in range(1, len(id2meta))]
        embeddings = semantic_encoder.encode(sorted_meta)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        np.save(meta_path, embeddings)
        print(f"Saved new embeddings to: {meta_path}")

    return embeddings.shape[-1]