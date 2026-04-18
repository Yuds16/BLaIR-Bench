# blair/cf/model/AlphaRec.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender

class AlphaRec(GeneralRecommender):
    """
    An example RecBole-adapted AlphaRec model.
    Note that this version inherits from RecBole's GeneralRecommender to comply with its configuration
    and training loop, rather than directly subclassing nn.Module.
    """
    def __init__(self, config, dataset):
        # Initialize using RecBole's config and dataset.
        super(AlphaRec, self).__init__(config, dataset)

        # Retrieve configuration parameters
        self.tau = config['tau']
        self.embed_size = config['embed_size']
        self.lm_model = config['semantic_encoder']  # This could be the identifier or an encoder instance.
        
        # for full‐sort caching
        self._cached_user_emb = None   # will hold [num_users, D]
        self._cached_item_emb = None   # will hold [num_items, D]
        
        self.init_item_cf_embeds = torch.tensor(dataset.item_cf_embeds, dtype=torch.float32, device=self.device)
        
        # 1) pull out NumPy array of item embeddings
        item_embs = dataset.item_cf_embeds # shape: (num_items, embed_dim)

        # 2) get train‐split user/item pairs
        train_users = dataset.inter_feat[self.USER_ID].numpy()
        train_items = dataset.inter_feat[self.ITEM_ID].numpy()

        # 3) build a list of items per user
        user2items = [[] for _ in range(dataset.user_num)]
        for u, i in zip(train_users, train_items):
            user2items[int(u)].append(int(i))
        
        # 4) average
        user_cf_embs = np.zeros((dataset.user_num, item_embs.shape[1]),
                                 dtype=item_embs.dtype)
        for u, items in enumerate(user2items):
            if items:  # if the user has any training interactions
                user_cf_embs[u] = np.mean(item_embs[items], axis=0)

        # 5) wrap as a frozen tensor
        self.init_user_cf_embeds = torch.tensor(
            user_cf_embs, dtype=torch.float32, device=self.device
        )

        self.init_embed_shape = self.init_user_cf_embeds.shape[1]
        
        # A simple linear transformation as in your original mlp (without nonlinearity)
        self.mlp = nn.Linear(self.init_embed_shape, self.embed_size, bias=False)

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self._cached_user_emb is not None or self._cached_item_emb is not None:
            self._cached_user_emb, self._cached_item_emb = None, None

        users     = interaction[self.USER_ID]     # [B]
        pos_items = interaction[self.ITEM_ID]     # [B]
        B = users.size(0)

        # 1) compute & normalize all embeddings once
        norm_all_user_emb = F.normalize(self.mlp(self.init_user_cf_embeds), dim=-1)  # [U, D]
        norm_all_item_emb = F.normalize(self.mlp(self.init_item_cf_embeds), dim=-1)  # [I, D]

        # 2) pick out this batch’s user & positive embeddings
        curr_user_emb = norm_all_user_emb[users]     # [B, D]
        curr_pos_item_emb = norm_all_item_emb[pos_items] # [B, D]

        # 3) Compute in batch user-item similarity matrix [B, B]
        logits = torch.matmul(curr_user_emb, curr_pos_item_emb.transpose(0, 1))
        logits = logits / self.tau

        # 4) Assign targets for in-batch contrastive learning
        targets = torch.arange(B, device=logits.device) 

        # 5) Calculate InfoNCE loss
        loss = F.cross_entropy(logits, targets)

        return loss
    
    def predict(self, interaction):
        users = interaction[self.USER_ID]      # e.g. tensor([u1,u2,...])
        items = interaction[self.ITEM_ID]      # e.g. tensor([i1,i2,...])
        
        all_users = self.mlp(self.init_user_cf_embeds)
        all_items = self.mlp(self.init_item_cf_embeds)
        
        u_emb = F.normalize(all_users[users], dim=-1)
        i_emb = F.normalize(all_items[items], dim=-1)
        return torch.sum(u_emb * i_emb, dim=-1)  # shape: [batch_size]

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        """
        Full-sort prediction method for ranking all items for each user.
        This method is invoked during evaluation.
        """
        # Extract user indices from the interaction object (e.g., a tensor)
        users = interaction[self.USER_ID]  # Expected shape: [batch_size]
        
        # 1) on first call, build & cache the per‐user and per‐item embeddings
        if self._cached_user_emb is None or self._cached_item_emb is None:
            # project your CF init embeddings
            all_users = self.mlp(self.init_user_cf_embeds)  # [U, D]
            all_items = self.mlp(self.init_item_cf_embeds)  # [I, D]

            # normalize once
            self._cached_user_emb = F.normalize(all_users, dim=-1)
            self._cached_item_emb = F.normalize(all_items, dim=-1)

        # 2) lookup only the users in this batch
        batch_users = self._cached_user_emb[users]        # [B, D]

        # 3) score against every item
        scores = batch_users @ self._cached_item_emb.t()  # [B, I]
        
        return scores