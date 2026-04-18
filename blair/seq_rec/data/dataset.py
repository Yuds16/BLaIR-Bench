import os
import json
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset


class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        # Try metadata cache first (for shared embeddings), then fall back to data_path
        feat_path_metadata = os.path.join("cache", "metadata", self.dataset_name, f'{self.dataset_name}.{self.plm_suffix}')
        feat_path_local = os.path.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        
        if os.path.exists(feat_path_metadata):
            feat_path = feat_path_metadata
        else:
            feat_path = feat_path_local
            
        loaded_feat = np.load(feat_path).reshape(-1, self.plm_size)
        assert loaded_feat.shape == (self.item_num - 1, self.plm_size), f"Loaded feature shape {loaded_feat.shape} does not match expected shape {(self.item_num - 1, self.plm_size)}"

        data_maps_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.data_maps')
        with open(data_maps_path, 'r') as f:
            data_maps = json.load(f)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(data_maps['item2id'][token]) - 1]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding
