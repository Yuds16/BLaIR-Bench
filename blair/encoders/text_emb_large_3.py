import numpy as np
import os
import tiktoken
from tqdm import tqdm
from openai import OpenAI

from blair.encoders.base import BaseSemanticEncoder

class TextEmbEncoder(BaseSemanticEncoder):
    """
    A semantic encoder that uses OpenAI's API for generating text embeddings.
    """
    def __init__(self, model = "text-embedding-3-large", batch_size=2048, api_key=None, **kwargs):
        # We pass a dummy gpu_id to the parent class for compatibility.
        # OpenAI models run on their servers, so local GPU is not used.
        super().__init__(model=model, gpu_id=None, batch_size=batch_size, **kwargs)

        # Store embedding dimensions parameter
        self.dims = kwargs.get("dims", None)
        self.max_tokens = 8192
        self.truncation_strategy = "truncate"

        # Initialize the OpenAI client.
        # It's best practice to set the OPENAI_API_KEY environment variable.
        # However, you can also pass the key directly if needed.
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if api_key is None:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or pass it directly.")

        self.client = OpenAI(
            api_key=api_key,  # Uncomment this line if you want to pass the API key
        )

        self.tokenizer = self._init_tokenizer()
    
    @property
    def name(self):
        """Returns the name of the model being used."""
        if self.dims != None:  # 3072 is the default dimension
            return f"{self.model}-mrl-d{self.dims}"
        return self.model

    def _init_tokenizer(self):
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def _prepare_input(self, text):
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_tokens:
            return text

        if self.truncation_strategy == "truncate":
            truncated_text = self.tokenizer.decode(tokens[:self.max_tokens])
            return truncated_text

        if self.truncation_strategy == "raise":
            raise ValueError(
                f"Input text exceeds the maximum of {self.max_tokens} tokens; received {len(tokens)} tokens."
            )

        raise ValueError(f"Unsupported truncation_strategy '{self.truncation_strategy}'.")

    def encode(self, sentences, **kwargs):
        """
        Generates embeddings for a list of sentences using the OpenAI API.

        Args:
            sentences (list[str]): A list of sentences to encode.

        Returns:
            np.ndarray: A NumPy array containing the embeddings.
        """
        all_embeddings = []
        
        # Process sentences in batches to stay within API limits and manage memory
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Encoding sentences"):
            batch = sentences[i:i + self.batch_size]
            processed_batch = []

            for j, text in enumerate(batch):
                prepared_text = self._prepare_input(text)
                processed_batch.append(prepared_text)
            
            try:
                # Make the API call to the OpenAI embeddings endpoint
                params = {
                    "input": processed_batch,
                    "model": self.model
                }
                if self.dims is not None:
                    params["dimensions"] = self.dims

                response = self.client.embeddings.create(**params)
                
                # Extract the embedding data from the response
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"An error occurred while encoding a batch: {e}")
                # Add placeholder embeddings (e.g., zeros) for the failed batch
                # to maintain array shape, or handle the error as needed.
                all_embeddings.extend(np.zeros((len(batch), self.dims or 3072)).tolist())

        return np.array(all_embeddings)