# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:32:14 2023

@author: junta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from paae import PAAE_layer
from aggregators import AttentionAggregator, MeanPoolAggregator

class DREAGR(nn.Module):
    """
    DEAGR for Group Recommendation:
    """
    def __init__(self, num_users, num_items, embedding_dim, drop_ratio, aggregator_type='attention'):
        super(DREAGR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.drop_ratio = drop_ratio
        self.aggregator_type = aggregator_type
        
        self.paee = PAAE_layer(self.num_items, self.num_users, self.embedding_dim, self.drop_ratio)
        
        if self.aggregator_type == 'meanpool':
            self.preference_aggregator = MeanPoolAggregator(self.embedding_dim, self.embedding_dim, self.drop_ratio)
        elif self.aggregator_type == 'attention':
            self.preference_aggregator = AttentionAggregator(self.embedding_dim, self.embedding_dim, self.drop_ratio)
        else:
            raise NotImplementedError("Aggregator type {} not implemented ".format(self.aggregator_type))
        
        self.group_predictor = nn.Linear(self.embedding_dim, self.num_items, bias=False)
        nn.init.xavier_uniform_(self.group_predictor.weight)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                
    def forward(self, group, group_mp_info, group_pmp_info, type_info = 'train'):
        """
        group_mp_info: group meta path
        group_pmp_info: group prerequisite meta path
        """
        if type_info == 'train':
            user_id_mp, group_mask_mp, group_items_mp, user_items_mp, _ = group_mp_info
            user_id_pmp, group_mask_pmp, group_items_pmp, user_items_pmp, _ = group_pmp_info
            
            user_embeds_fusion, _ = self.paee(user_items_mp, user_items_pmp, type_info = 'group')
            group_embeds = self.preference_aggregator(user_embeds_fusion, (group_mask_mp+group_mask_pmp), mlp=True)
            
            group_logits = self.group_predictor(group_embeds)
        elif type_info == 'test':
            _, group_mask_mp, group_items_mp, user_items_mp = group_mp_info
            _, group_mask_pmp, group_items_pmp, user_items_pmp = group_pmp_info
            
            user_embeds_fusion, _ = self.paee(user_items_mp, user_items_pmp, type_info = 'group')
            group_embeds = self.preference_aggregator(user_embeds_fusion, (group_mask_mp+group_mask_pmp), mlp=True)
            group_logits = self.group_predictor(group_embeds)
        return group_logits, group_embeds
        

    def multinomial_loss(self, logits, items):
        """ multinomial likelihood with softmax over item set """
        return -torch.mean(torch.sum(F.log_softmax(logits, 1) * items, -1))

    def user_loss(self, user_logits, user_items):
        return self.multinomial_loss(user_logits, user_items)

    def loss(self, group_logits, group_items, user_items, group_mask):
        ui_sum = user_items.sum(2, keepdim=True)  # [B, G]
        user_items_norm = user_items / torch.max(torch.ones_like(ui_sum), ui_sum)  # [B, G, I]
        gi_sum = group_items.sum(1, keepdim=True)
        group_items_norm = group_items / torch.max(torch.ones_like(gi_sum), gi_sum)  # [B, I]
        group_mask_zeros = torch.exp(group_mask).unsqueeze(2)  # [B, G, 1]
        
        user_items_norm = torch.sum(user_items_norm * group_mask_zeros, dim=1) / group_mask_zeros.sum(1)
        user_group_loss = self.multinomial_loss(group_logits, user_items_norm)
        group_loss = self.multinomial_loss(group_logits, group_items_norm)

        return group_loss + user_group_loss






