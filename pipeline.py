# pipeline.py

import argparse 
import ast
from blair.blair import BLaIRBenchmark
from blair.utils import init_semantic_encoder

def pipeline(
        plm: str, 
        task: str, 
        datasets: list[str], 
        gpu_id: int,
        encoder_kwargs: dict[str, any]={"emb_type":'CLS'}, 
        hyperdict: dict[str, any]={"learning_rate": [3e-3 , 1e-3, 3e-4]},
        enc_bs=256, 
        eval_bs=128,
        features_needed: list=['title'], 
    ):

    # Exceptions handling:
    task2ds = {
        "seq_rec": ["Beauty_and_Personal_Care", "All_Beauty", "Video_Games", "Baby_Products", "ML-1M", "Yelp"],
        "cf": ["Beauty_and_Personal_Care", "All_Beauty", "Video_Games", "Baby_Products", "ML-1M", "Yelp"],
        "prod_search": ["Amazon-C4", "esci", "reddit_movie"]
    }
    # If a task is not valid
    if task not in task2ds:
        raise ValueError("Task not supported by BLaIR-Bench")
    
    # If any ds is not supported
    unsupported = [ds for ds in datasets if ds not in task2ds[task]]
    if len(unsupported) > 0:
        raise ValueError(f"Datasets {unsupported} not supported by BLaIR-Bench")

    # If datasets is empty, test all ds under that task
    if len(datasets) == 0:
        datasets = task2ds[task]

    semantic_encoder = init_semantic_encoder(
        model=plm, 
        gpu_id=gpu_id,
        batch_size=enc_bs,
        **encoder_kwargs
    )

    benchmark = BLaIRBenchmark(
        # seq_rec / prod_search / cf
        task=task,
        # prod_search: "Amazon-C4", "esci" 
        # seq_rec: "All_Beauty", "Video_Games", "Baby_Products", "ML-1M", "Yelp"
        # cf: "All_Beauty", "Video_Games", "Baby_Products", "Book-Crossing", "ML-1M", "Yelp"
        datasets=datasets, 
        gpu_id=gpu_id,
        cache_path="cache",
        eval_batch_size=eval_bs,
        features_needed=features_needed
    )

    results = benchmark.run(
        semantic_encoder,
        output_folder="results",
        gpu_id=gpu_id,
        hyperdict=hyperdict
    )
    print("Test results:", results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BLaIR-Bench pipeline")
    parser.add_argument("--model",        type=str,   default="bert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--emb-type",     choices=["CLS","Mean"], default="CLS",
                        help="Embedding pooling strategy")
    parser.add_argument("--task",         choices=["seq_rec","cf","prod_search"],
                        required=True, help="Which task to run")
    parser.add_argument("--datasets",     nargs="*",  default=[],
                        help="(Optional) List of datasets; omit to run all")
    parser.add_argument("--gpu_id",       type=int,   default=0,
                        help="CUDA device ID (-1 for CPU)")
    parser.add_argument("--enc_bs",       type=int,   default=64,
                        help="Batch size for encoding")
    parser.add_argument("--eval_bs",      type=int,   default=128,
                        help="Batch size for evaluation")
    parser.add_argument("--pca", action="store_true",
                        help="Enable PCA")
    parser.add_argument("--whiten", action="store_true",
                        help="Enable PCA whitening")
    parser.add_argument("--n_comps",      type=int,   default=0,
                        help="Number of components for PCA")
    parser.add_argument("--features", nargs='+', default=['title'],
                        help="List of features to use for metadata cleaning")
    parser.add_argument("--dims", type=int, default=None,
                        help="Embedding dimensions for API-based encoders (e.g., OpenAI, Gemini) and Qwen3. If not set, use the model default.")
    parser.add_argument(
        "--hyperdict",
        type=ast.literal_eval,
        default='{"learning_rate": [3e-3, 1e-3, 3e-4]}',
        help="A Python‐literal dict of hyperparameters, e.g. "
             "'{\"learning_rate\": [1e-3, 1e-4]}'"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # build the encoder_kwargs dict
    encoder_kwargs = {
        "emb_type":  args.emb_type,
        "pca": args.pca,
        "n_comps": args.n_comps,
        "whiten": args.whiten,
    }

    # Add dims parameter if specified
    if args.dims is not None:
        encoder_kwargs["dims"] = args.dims

    pipeline(
        plm            = args.model,
        task           = args.task,
        datasets       = args.datasets,
        gpu_id         = args.gpu_id,
        encoder_kwargs = encoder_kwargs,
        hyperdict      = args.hyperdict,
        enc_bs         = args.enc_bs,
        eval_bs        = args.eval_bs,
        features_needed= args.features,
    )

if __name__ == "__main__":
    main()
