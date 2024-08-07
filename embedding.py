import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding_user(nn.Module):
    """
        user-item(n, m) --> (n, embedding_dim)
    """
    def __init__(self, num_items, embedding_dim):
        super(Embedding_user, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        #元路径
        self.user_embedding_path = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip([self.num_items], [self.embedding_dim])):
            layer = torch.nn.Linear(in_size, out_size, bias=True)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.user_embedding_path.append(layer)
        
    def forward(self, user_item_inputs):
        """ user embedding encoder
        """
        user_embedding = F.normalize(user_item_inputs)
        for idx, _ in enumerate(range(len(self.user_embedding_path))):
            user_embedding = self.user_embedding_path[idx](user_embedding)  # [B, G, D] or [B, D]
            user_embedding = torch.tanh(user_embedding)
            
        return user_embedding


class Embedding_item(nn.Module):
    """
    item-user(n, m) --> (n, embedding_dim)
    PAAE for item embedding on mp and pmp:
    """
    def __init__(self, num_users, embedding_dim):
        super(Embedding_item, self).__init__()
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        
        #元路径
        self.item_embedding_path = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip([self.num_users], [self.embedding_dim])):
            layer = torch.nn.Linear(in_size, out_size, bias=True)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.item_embedding_path.append(layer)
    
    def forward(self, item_user_inputs):
        """ item embedding encoder
        """
        item_embedding = F.normalize(item_user_inputs)
        for idx, _ in enumerate(range(len(self.item_embedding_path))):
            item_embedding = self.item_embedding_path[idx](item_embedding)  # [B, G, D] or [B, D]
            item_embedding = torch.tanh(item_embedding)
        
        return item_embedding

