# blair/prod_search/task.py

import os
import json
import numpy as np

from blair.prod_search.generate_emb import generate_item_emb, generate_query_emb
from blair.prod_search.eval_search import evaluate_search
from blair.utils import init_device


class ProdSearchBaseTask:
    def __init__(self, dataset_name, gpu_id=0, cache_path = "./cache", eval_batch_size=64, **kwargs):
        self.dataset_name = dataset_name
        self.device = init_device(gpu_id=gpu_id)
        self.cache_path = os.path.join(cache_path, "prod_search")
        self.eval_batch_size = eval_batch_size

        self.results = {}
    
    def load_data(self, semantic_encoder):
        """
        Generate item & query embeddings
        """
        processed_item_dir = os.path.join(self.cache_path, self.dataset_name, f'{self.dataset_name}.{semantic_encoder.name}.npy')
        if not os.path.exists(processed_item_dir):
            print(f"[ProdSearchBaseTask] Processing item data for {self.dataset_name} ...")
            # Generate item embeddings
            emb_size_item = generate_item_emb(
                dataset_name=self.dataset_name,
                cache_path=self.cache_path,
                semantic_encoder=semantic_encoder
            )
            self.emb_size_item = emb_size_item
        else:
            print(f"[ProdSearchBaseTask] Data for {self.dataset_name} is already processed.")
            features = np.load(processed_item_dir)
            print(features.shape)
            self.emb_size_item = features.shape[-1]

        processed_query_dir = os.path.join(self.cache_path, self.dataset_name, f'{self.dataset_name}.q_{semantic_encoder.name}.npy')
        if not os.path.exists(processed_query_dir):
            print(f"[ProdSearchBaseTask] Processing query data for {self.dataset_name} ...")
            # Generate query embeddings
            emb_size_query = generate_query_emb(
                dataset_name=self.dataset_name,
                cache_path=self.cache_path,
                semantic_encoder=semantic_encoder
            )
            self.emb_size_query = emb_size_query
        else:
            print(f"[ProdSearchBaseTask] Data for {self.dataset_name} is already processed.")
            features = np.load(processed_query_dir)
            print(features.shape)
            self.emb_size_query = features.shape[-1]
        
        if self.emb_size_item == self.emb_size_query:
            self.emb_size = self.emb_size_item
        else:
            raise ValueError(
                f"Item embedding size ({emb_size_item}) does not match query embedding size ({emb_size_query})."
            )
    
    def run(self, semantic_encoder, gpu_id, **kwargs):
        """
        1) Evaluate with eval_search.py
        2) Optionally store the results in self.results
        """

        self.model_name = semantic_encoder.name

        # If you'd like to store the returned metrics, modify evaluate_search to return them
        self.results = evaluate_search(
            dataset=self.dataset_name,
            suffix=semantic_encoder.name + '.npy',
            k=100,
            gpu_id=gpu_id,
            batch_size=self.eval_batch_size,
            domain=False,           # or True if you want domain-level results
            data_path=self.cache_path,
            emb_size=self.emb_size
        )

        # For now, I set a placeholder
        # self.results["status"] = f"Evaluation finished for {self.dataset_name}"

    def save_results(self, output_folder="results"):
        """
        Save self.results to a JSON file, or you can customize.
        """
        os.makedirs(output_folder, exist_ok=True)
        out_file = os.path.join(output_folder, f"prodsearch_{self.model_name}_{self.dataset_name}_results.json")
        with open(out_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[ProdSearchBaseTask] Results saved to: {out_file}")

    