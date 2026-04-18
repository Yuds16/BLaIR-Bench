import os
import time
import json
import numpy as np
from tqdm import tqdm
import google.generativeai as genai

from blair.encoders.base import BaseSemanticEncoder

class GeminiEmbEncoder(BaseSemanticEncoder):
    """
    A semantic encoder that uses Google's Gemini API for generating text embeddings.
    """
    def __init__(self, model="gemini-embedding-001", batch_size=1024, api_key=None, **kwargs):
        # Pass dummy gpu_id to parent; Gemini is API-based
        super().__init__(model=model, gpu_id=None, batch_size=batch_size, **kwargs)

        # Store embedding dimensions parameter
        self.dims = kwargs.get("dims", None)

        # Progress tracking
        api_log_dir = "api_log"
        os.makedirs(api_log_dir, exist_ok=True)
        self.progress_file = kwargs.get("progress_file", os.path.join(api_log_dir, f"{model}_progress.json"))
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 50000)  # Save every 50000 batches

        # Configure the Google client with the API key
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable or pass it directly.")

        genai.configure(
            api_key=api_key # Uncomment this line if you want to pass the API key
        )

    @property
    def name(self):
        """Returns the name of the Gemini model being used."""
        if self.dims != None:  # 3072 is the default dimension
            return f"{self.model}-mrl-d{self.dims}"
        return self.model

    def _load_progress(self):
        """Load progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {"completed_batches": [], "embeddings": []}
        return {"completed_batches": [], "embeddings": []}

    def _save_progress(self, progress):
        """Save progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)

    def _encode_batch_with_retry(self, batch, batch_idx, max_retries=3):
        """Encode a batch with retry logic."""
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=batch,
                    output_dimensionality=self.dims
                )
                return result['embedding']
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Batch {batch_idx} attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Batch {batch_idx} failed after {max_retries} attempts, using zero embeddings")
                    return np.zeros((len(batch), self.dims or 3072)).tolist()
        return None

    def encode(self, sentences, **kwargs):
        """
        Generates embeddings for a list of sentences using the Gemini API with retry logic and progress tracking.
        """
        # Load existing progress
        progress = self._load_progress()
        completed_batches = set(progress.get("completed_batches", []))
        all_embeddings = progress.get("embeddings", [])

        # Calculate total number of batches
        total_batches = (len(sentences) + self.batch_size - 1) // self.batch_size

        # If resuming, find the last completed position
        start_batch = len(completed_batches)
        if start_batch > 0:
            print(f"Resuming from batch {start_batch}/{total_batches} (completed: {len(completed_batches)} batches)")

        # Process sentences in batches
        for batch_idx in tqdm(range(start_batch, total_batches), desc="Encoding sentences", initial=start_batch, total=total_batches):
            # Skip if batch already completed
            if batch_idx in completed_batches:
                continue

            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(sentences))
            batch = sentences[start_idx:end_idx]

            # Encode batch with retry logic
            batch_embeddings = self._encode_batch_with_retry(batch, batch_idx)
            all_embeddings.extend(batch_embeddings)

            # Mark batch as completed
            completed_batches.add(batch_idx)

            # Save progress periodically
            if (batch_idx + 1) % self.checkpoint_interval == 0:
                progress = {
                    "completed_batches": list(completed_batches),
                    "embeddings": all_embeddings
                }
                self._save_progress(progress)
                print(f"Progress saved: {len(completed_batches)}/{total_batches} batches completed")

        # Final save
        progress = {
            "completed_batches": list(completed_batches),
            "embeddings": all_embeddings
        }
        self._save_progress(progress)
        print(f"Encoding completed: {len(completed_batches)}/{total_batches} batches")

        return np.array(all_embeddings)