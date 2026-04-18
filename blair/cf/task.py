# blair/cf/task.py

import os
import json
import numpy as np
import time
from tqdm import tqdm

from blair.utils import init_device, apply_pca_if_needed
from blair.cf.dataset.process_amazon_2023 import process_amazon_cf
from blair.cf.dataset.process_yelp import process_yelp_cf
from blair.cf.dataset.process_bc import process_bc_cf
from blair.cf.dataset.process_ml import process_ml_cf
from blair.cf.dataset.process_cf_test import process_cf_test
from blair.cf.run import run_single


class HyperParamLoader:
    """
    Helper class: generates all hyperparameter combinations from a dictionary (arg_range)
    using a DFS-based approach.
    """
    def __init__(self, arg_range):
        self.arg_range = arg_range
        self.k_list = list(arg_range.keys())
        self.choice = np.zeros((len(self.k_list)), dtype=int)
        self.args = []
        self._dfs(0)
        # Sanity-check: total combinations equals product of all list lengths
        assert len(self.args) == np.prod([len(v) for v in arg_range.values()])
        self.n_args = len(self.args)
        print('Total hyperparameter combinations:', self.n_args, flush=True)
    
    def _dfs(self, layer):
        if layer == len(self.k_list):
            ans = {}
            for l, k in enumerate(self.k_list):
                ans[k] = self.arg_range[k][self.choice[l]]
            self.args.append(ans)
            return
        
        k = self.k_list[layer]
        for i in range(len(self.arg_range[k])):
            self.choice[layer] = i
            self._dfs(layer + 1)


class CFBaseTask:
    def __init__(self, dataset_name, gpu_id=0, cache_path="./cache", enc_batch_size=32, eval_batch_size=8, **kwargs):
        self.dataset_name = dataset_name
        self.results = {}
        self.cache_path = os.path.join(cache_path, "cf")
        self.enc_batch_size = enc_batch_size
        self.eval_batch_size = eval_batch_size
        self.device = init_device(gpu_id)
        self.features_needed = kwargs.get('features_needed', ['title'])
    
    def load_data(self, semantic_encoder=None):
        amazon2023 = ['All_Beauty', 'Video_Games', 'Baby_Products']

        # Check if task-specific data files (.inter files and data_maps) exist
        task_data_dir = os.path.join(self.cache_path, self.dataset_name)
        data_maps_file = os.path.join(task_data_dir, f"{self.dataset_name}.data_maps")
        train_inter_file = os.path.join(task_data_dir, f"{self.dataset_name}.train.inter")

        # Check if embedding files exist
        processed_dir_metadata = os.path.join("cache", "metadata", self.dataset_name, f'{self.dataset_name}.{semantic_encoder.name}.npy')
        processed_dir_local = os.path.join(self.cache_path, self.dataset_name, f'{self.dataset_name}.{semantic_encoder.name}.npy')
        embeddings_exist = os.path.exists(processed_dir_metadata) or os.path.exists(processed_dir_local)

        # Always process data if task-specific files or embeddings don't exist
        if not (os.path.exists(data_maps_file) and os.path.exists(train_inter_file) and embeddings_exist):
            print(f"[CFBaseTask] Processing data for {self.dataset_name} ...")
            if self.dataset_name in amazon2023:
                # Now call process_amazon(...) directly
                emb_size = process_amazon_cf(
                    domain=self.dataset_name,
                    device=self.device,
                    semantic_encoder=semantic_encoder,
                    output_dir=self.cache_path,
                    batch_size=self.enc_batch_size,
                    features_needed=self.features_needed
                    # any other arguments like n_workers=16, batch_size=16, etc.
                )
            elif self.dataset_name == 'testCF':
                emb_size = process_cf_test(
                    device=self.device,
                    semantic_encoder=semantic_encoder,
                    output_dir=self.cache_path,
                    batch_size=self.enc_batch_size
                )
            elif self.dataset_name == "Yelp":
                emb_size = process_yelp_cf(
                    semantic_encoder=semantic_encoder,
                    output_dir=self.cache_path,
                    yelp_config={"metadata": "sentence",  # none / raw / sentence
                    "split": "last_out", # last_out / timestamp
                    "user_core": 5,
                    "item_core": 5,
                    "rating_score": 0.0, # rating score smaller than this score would be deleted
                    "version": "Yelp_2020",  # ["Yelp_2018", "Yelp_2020", "Yelp_2021", "Yelp_2022"]
                    "date_min": "2019-01-01 00:00:00",
                    "date_max": "2019-12-31 00:00:00"}
                )
            elif self.dataset_name == "Book-Crossing":
                emb_size = process_bc_cf(
                    semantic_encoder=semantic_encoder,
                    output_dir=self.cache_path
                )
            elif self.dataset_name == "ML-1M":
                emb_size = process_ml_cf(
                    semantic_encoder=semantic_encoder,
                    output_dir=self.cache_path
                )
            else:
                '''
                Space for ML-1M and Book-Crossing
                '''
                assert NotImplementedError()
        else:
            print(f"[CFBaseTask] Data for {self.dataset_name} is already processed.")

        # Find and load embeddings - try metadata cache first, then data cache
        if os.path.exists(processed_dir_metadata):
            processed_dir = processed_dir_metadata
        else:
            processed_dir = processed_dir_local

        features = np.load(processed_dir)
        print(f"Loaded raw embeddings shape: {features.shape}")

        # Apply PCA post-processing if needed
        pca_config = semantic_encoder.get_pca_config()
        features, processed_emb_path = apply_pca_if_needed(
            features,
            semantic_encoder.name,
            self.dataset_name,
            pca_config
        )
        print(f"Final embeddings shape after PCA: {features.shape}")

        self.emb_size = features.shape[-1]
        # Store processed emb path for training
        self.processed_emb_path = processed_emb_path
    
    def run(self, semantic_encoder, gpu_id=0, hyperdict={"learning_rate": [3e-3, 1e-3, 3e-4]}):
        """
        Performs hyperparameter tuning by iterating over all combinations provided
        in hyperdict. For each combination, it calls run_single() and logs progress.
        The same log file (self.log_file) will be used as the final result file.
        """
        self.model_name = semantic_encoder.name
        print(f"[CFBaseTask] Starting hyperparameter tuning with model={semantic_encoder.name}, domain={self.dataset_name}")
        
        # Setup logging: create 'hyper_cf' folder and log file.
        log_dir = "hyper_cf"
        os.makedirs(log_dir, exist_ok=True)
        # Generate a log file name based on sys.argv
        self.log_file = os.path.join(log_dir, f"log_{int(time.time())}.txt")
        hlog = open(self.log_file, 'a+')
        
        # Write meta information
        hlog.write("======= META =======\n")
        meta_info = {
            "model_name": semantic_encoder.name,
            "domain_name": self.dataset_name,
            "hyperdict": hyperdict
        }
        hlog.write(str(meta_info) + "\n\n")
        hlog.flush()
        
        # Base configuration to be passed to run_single()
        pca_config = semantic_encoder.get_pca_config()
        suffix = self.processed_emb_path.rsplit('/', 1)[-1]
        suffix = '.'.join(suffix.split('.')[1:])

        base_config = {
            "gpu_id": gpu_id,
            "data_path": self.cache_path,
            "device": self.device,
            "plm_size": self.emb_size,
            "plm_suffix": suffix,
            "semantic_encoder": semantic_encoder.name.split("/")[-1]
        }
        
        # Instantiate hyperparameter loader to generate all combinations
        hp_loader = HyperParamLoader(hyperdict)
        
        best_valid_score = None
        best_params = None
        best_result_dict = None
        best_round = None
        
        # Iterate over each hyperparameter combination
        for idx, hyper_params in enumerate(tqdm(hp_loader.args, desc="Hyperparameter Tuning")):
            hlog.write(f"======= Round {idx} =======\n")
            hlog.write(str(hyper_params) + "\n\n")
            hlog.flush()
            
            # Merge base configuration with current hyperparameter combination
            config_overrides = base_config.copy()
            config_overrides.update(hyper_params)
            
            print(f"[CFBaseTask] Tuning round {idx+1}/{hp_loader.n_args} with hyperparameters: {hyper_params}")
            # Call run_single() with the current configuration
            _, _, result_dict = run_single(
                model_name="AlphaRec",
                dataset=self.dataset_name,
                pretrained_file="",
                **config_overrides
            )
            
            hlog.write("Best Valid Result: " + str(result_dict["best_valid_result"]) + "\n\n")
            hlog.flush()
            
            current_valid_score = result_dict["best_valid_score"]
            print(f"[CFBaseTask] Round {idx+1}: Best Valid Score = {current_valid_score}")
            
            # Update the best configuration if current run is better
            if best_valid_score is None or current_valid_score > best_valid_score:
                hlog.write(f"\n Best Valid Updated: {best_valid_score} -> {current_valid_score}\n\n")
                hlog.flush()
                best_valid_score = current_valid_score
                best_params = hyper_params
                best_result_dict = result_dict
                best_round = idx
        
        # Write final summary
        hlog.write("======= FINAL =======\n")
        hlog.write(f"Best Round: {best_round}\n")
        hlog.write(f"Best Valid Score: {best_valid_score}\n")
        hlog.write("Best Params: " + str(best_params) + "\n")
        hlog.write("Final Valid Result: " + str(best_result_dict.get("best_valid_result", "")) + "\n")
        hlog.write("Final Test Result: " + str(best_result_dict.get("test_result", "")) + "\n")
        hlog.close()
        
        # Save the best results into the task's results (if needed for other purposes)
        self.results["best_valid_score"] = best_valid_score
        self.results["best_params"] = best_params
        if best_result_dict:
            self.results["best_valid_result"] = best_result_dict["best_valid_result"]
            self.results["test_result"] = best_result_dict["test_result"]
        
        print("[CFBaseTask] Hyperparameter tuning completed.")
        print(f"Best Valid Score: {best_valid_score}")
        print(f"Best Hyperparameters: {best_params}")

    def save_results(self, output_folder: str):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        result_file = os.path.join(output_folder, f"cf_{self.model_name}_{self.dataset_name}_results.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
            
    def __str__(self):
        return f"CFTask(domain={self.dataset_name})"