# tests/task/test_tasks.py

import pytest
import sys
import os
import tempfile
cache_dir = tempfile.mkdtemp()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from blair.blair import BLaIRBenchmark
from blair.utils import init_semantic_encoder

def test_small_prod_search():
    """
    An example 'smoke test' that checks if prod_search task can run on a
    small dataset without errors.
    """
    gpu_id = 0
    semantic_encoder = init_semantic_encoder(
        model='roberta-base',
        gpu_id=gpu_id,
        batch_size=32,
        emb_type='CLS'
    )

    benchmark = BLaIRBenchmark(
        task='prod_search',
        datasets=['testProdSearch'],
        gpu_id=gpu_id,
        cache_path=cache_dir,
        eval_batch_size=64
    )

    results = benchmark.run(
        semantic_encoder,
        output_folder="results",
        gpu_id=gpu_id
    )

def test_small_seq_rec():
    """
    An example 'smoke test' that checks if seq_rec task can run on a
    small dataset without errors.
    """
    gpu_id = 0
    semantic_encoder = init_semantic_encoder(
        model='roberta-base',
        gpu_id=gpu_id,
        batch_size=32,
        emb_type='CLS'
    )

    benchmark = BLaIRBenchmark(
        task='seq_rec',
        datasets=['testSeqRec'],
        gpu_id=gpu_id,
        cache_path=cache_dir,
        eval_batch_size=64
    )

    results = benchmark.run(
        semantic_encoder,
        output_folder="results",
        gpu_id=gpu_id,
        hyperdict = {
            "epoch": [1]
        }
    )

def test_small_cf():
    """
    An example 'smoke test' that checks if cf task can run on a
    small dataset without errors.
    """
    gpu_id = 0
    semantic_encoder = init_semantic_encoder(
        model='roberta-base',
        gpu_id=gpu_id,
        batch_size=32,
        emb_type='CLS'
    )

    benchmark = BLaIRBenchmark(
        task='cf',
        datasets=['testCF'],
        gpu_id=gpu_id,
        cache_path=cache_dir,
        eval_batch_size=64
    )

    results = benchmark.run(
        semantic_encoder,
        output_folder="results",
        gpu_id=gpu_id,
        hyperdict = {
            "epoch": [1],
            "topk": [1],        # only evaluate at @1
            "valid_metric": ["ndcg@1"]

        }
    )