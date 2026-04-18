# main.py

import argparse

from blair.blair import BLaIRBenchmark
from blair.utils import init_semantic_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="BLaIR Benchmark")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    semantic_encoder = init_semantic_encoder(
        model='princeton-nlp/sup-simcse-roberta-large', # 'bert-base-uncased'
        gpu_id=args.gpu_id,
        batch_size=64, 
        **{"emb_type":'CLS', 
           "pca": False,
           "n_comps": 0,
           "whiten": False}  # kwargs for the encoder
    )

    benchmark = BLaIRBenchmark(
        # seq_rec / prod_search / cf
        task="seq_rec", 
        datasets=["All_Beauty"],
        gpu_id=args.gpu_id,
        cache_path="cache",
        eval_batch_size=128,
        features_needed=['title'] 
    )

    results = benchmark.run(
        semantic_encoder,
        output_folder="results",
        gpu_id=args.gpu_id,
        hyperdict={
            "learning_rate": [3e-3 , 1e-3, 3e-4] 
        }
    )

    print("Test results:", results)


if __name__ == "__main__":
    main()
