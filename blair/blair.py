# blair/blair.py

from typing import List
from blair.seq_rec.task import SeqRecBaseTask
from blair.prod_search.task import ProdSearchBaseTask
from blair.cf.task import CFBaseTask

class BLaIRBenchmark:
    def __init__(
        self,
        task: str,
        datasets: List[str],
        gpu_id: int = 0,
        cache_path: str = "./cache",
        eval_batch_size: int = 8,
        features_needed=['title']
    ):
        """
        Initialize BLaIRBenchmark with a task type and dataset list.
        
        :param task: One of {'seq_rec', 'prod_search', 'cf'}.
        :param datasets: List of dataset names for the chosen task.
        :param gpu_id: Which GPU to use (default 0).
        :param cache_path: Path to cache directory.
        :param eval_batch_size: Batch size for evaluation.
        """
        self.results = {}

        # Create the tasks based on the user's parameters.
        if task == "seq_rec":
            name_to_class = {
                "Beauty_and_Personal_Care": BeautyPersonalCareTask,
                "All_Beauty": BeautyTask,
                "Video_Games": GamesTask,
                "Baby_Products": BabyTask,
                "ML-1M": MovieTask,
                "Yelp": YelpSqTask,
                "testSeqRec": testSeqRec,
                "testApi": testApi
            }
        elif task == "cf":
            name_to_class = {
                "Beauty_and_Personal_Care": CFBeautyPersonalCareTask,
                "All_Beauty": CFBeautyTask,
                "Video_Games": CFGamesTask,
                "Baby_Products": CFBabyTask,
                "Book-Crossing": CFBookTask,
                "ML-1M": CFMovieTask,
                "Yelp": YelpCfTask,
                "testCF": testCF,
            }
        elif task == "prod_search":
            name_to_class = {
                "esci": ESCITask,
                "Amazon-C4": C4Task,
                "testProdSearch": testProdSearch,
                "testApi": testApiPS,
                "reddit_movie": redditMovieTask
            }
        else:
            raise ValueError(f"Unknown task: {task}")

        # Build the tasks
        selected_tasks = []
        for d in datasets:
            if d not in name_to_class:
                raise ValueError(f"Dataset '{d}' not recognized for task '{task}'")

            # Pass task-specific kwargs if needed            
            task_specific_kwargs = {"features_needed": features_needed} if task in ['seq_rec', 'cf'] else {}

            # Instantiate the task class
            selected_tasks.append(
                name_to_class[d](
                    gpu_id=gpu_id,
                    cache_path=cache_path,
                    eval_batch_size=eval_batch_size,
                    **task_specific_kwargs
                )
            )

        # Store them for later usage in run()
        self.tasks = selected_tasks

    def run(self, semantic_encoder, output_folder="results", gpu_id=0, **kwargs):
        """
        For each task in self.tasks:
         1) task.load_data(semantic_enocder=...)
         2) task.run(semantic_enocder=..., gpu_id=..., **kwargs)
         3) task.save_results(output_folder=...)
         4) store results into self.results
        """
        for task in self.tasks:
            # 1) Load & process data
            task.load_data(semantic_encoder=semantic_encoder)

            # 2) Train/Eval
            task.run(
                semantic_encoder=semantic_encoder,
                gpu_id=gpu_id,
                **kwargs
            )

            # 3) Save results
            task.save_results(output_folder=output_folder)
            self.results[task.dataset_name] = task.results

        return self.results


# Unitest task
class testProdSearch(ProdSearchBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="testProdSearch", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

# Unitest task
class testApiPS(ProdSearchBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="testApi", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

# Unitest task
class testSeqRec(SeqRecBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="testSeqRec", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

# API cost task
class testApi(SeqRecBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="testApi", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

# CF datasets
class testCF(CFBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="testCF", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class CFBeautyPersonalCareTask(CFBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Beauty_and_Personal_Care", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class CFBeautyTask(CFBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="All_Beauty", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class CFGamesTask(CFBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Video_Games", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class CFBabyTask(CFBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Baby_Products", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class CFBookTask(CFBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Book-Crossing", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class CFMovieTask(CFBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="ML-1M", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class YelpCfTask(CFBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Yelp", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

# Product Search dataset
class ESCITask(ProdSearchBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="esci", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class C4Task(ProdSearchBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Amazon-C4", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class redditMovieTask(ProdSearchBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="reddit_movie", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

# SeqRec datasets
class BeautyPersonalCareTask(SeqRecBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Beauty_and_Personal_Care", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class BeautyTask(SeqRecBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="All_Beauty", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class GamesTask(SeqRecBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Video_Games", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class BabyTask(SeqRecBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Baby_Products", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class MovieTask(SeqRecBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="ML-1M", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)

class YelpSqTask(SeqRecBaseTask):
    def __init__(self, gpu_id=0, cache_path="./cache", eval_batch_size=8, **kwargs):
        super().__init__(dataset_name="Yelp", gpu_id=gpu_id, cache_path=cache_path, eval_batch_size=eval_batch_size, **kwargs)
