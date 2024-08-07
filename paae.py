import torch
import torch.nn as nn
from embedding import Embedding_user, Embedding_item

class PAAE_layer(nn.Module):
    def __init__(self, num_items, num_users, embedding_dim, drop_ratio):
        super(PAAE_layer, self).__init__()
        
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.drop_ratio = drop_ratio
        
        #meta-path--user embedding
        self.user_embedding_mp = Embedding_user(self.num_items, self.embedding_dim)
        #dependency meta-path--user embedding
        self.user_embedding_pmp = Embedding_user(self.num_items, self.embedding_dim)
        #Sigmoid
        self.MLP = nn.Sequential(
            nn.Linear(self.embedding_dim *2, self.embedding_dim, bias=True),
            nn.Sigmoid(),
            nn.Dropout(self.drop_ratio))
        
        self.transform_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_uniform_(self.transform_layer.weight)
        nn.init.zeros_(self.transform_layer.bias)
        
        self.user_predictor = nn.Linear(self.embedding_dim, self.num_items, bias=False)  # item embedding for pre-training
        nn.init.xavier_uniform_(self.user_predictor.weight)
    
    def paae_fusion(self, ui_input_mp, ui_input_pmp, type_info = 'user'):
        if type_info == 'user':
            user_embeds_mp = self.user_embedding_mp(ui_input_mp)
            item_input_mp = ui_input_mp.T
            
            user_embeds_pmp = self.user_embedding_pmp(ui_input_pmp)
            item_input_pmp = ui_input_pmp.T
            
            #meta-path attention
            self.item_embedding_mp = Embedding_item(ui_input_mp.shape[0], self.embedding_dim)
            item_embeds_mp = self.item_embedding_mp(item_input_mp)
            user_item_embed_mp = torch.cat((user_embeds_mp, item_embeds_mp))
            
            self.attentionLayer_mp = AttentionLayer(self.embedding_dim, item_embeds_mp.shape[0], user_item_embed_mp.shape[0], user_embeds_mp.shape[0], self.drop_ratio)
            at_mp = self.attentionLayer_mp(user_item_embed_mp)
            user_embeds_mp_at = torch.matmul(at_mp, item_embeds_mp)
            
            #dependency meta-path attention
            self.item_embedding_pmp = Embedding_item(ui_input_pmp.shape[0], self.embedding_dim)
            item_embeds_pmp = self.item_embedding_pmp(item_input_pmp)
            user_item_embed_pmp = torch.cat((user_embeds_pmp, item_embeds_pmp))
            
            self.attentionLayer_pmp = AttentionLayer(self.embedding_dim, item_embeds_pmp.shape[0], user_item_embed_pmp.shape[0], user_embeds_pmp.shape[0], self.drop_ratio)
            at_pmp = self.attentionLayer_pmp(user_item_embed_pmp)
            user_embeds_pmp_at = torch.matmul(at_pmp, item_embeds_pmp)
            
            #concat
            user_embeddings_mp = torch.cat((user_embeds_mp, user_embeds_mp_at), dim=1)
            user_embeddings_mp = self.MLP(user_embeddings_mp)
            user_embeddings_pmp = torch.cat((user_embeds_pmp, user_embeds_pmp_at), dim=1)
            user_embeddings_pmp = self.MLP(user_embeddings_pmp)
            #fusion
            fusion_rate = torch.sigmoid(torch.add(user_embeddings_mp, user_embeddings_pmp))
            user_embeds_fusion = torch.add(torch.mul(fusion_rate, user_embeddings_mp), torch.mul((1-fusion_rate), user_embeddings_pmp))
            
            #item
            item_embeddings = torch.cat((item_embeds_mp, item_embeds_pmp), dim=1)
            item_embeddings = self.MLP(item_embeddings)
            return user_embeds_fusion, item_embeddings
        
        elif type_info == 'group':
            user_embeds_mp = self.user_embedding_mp(ui_input_mp)
            item_input_mp = ui_input_mp.transpose(2,1)
            
            user_embeds_pmp = self.user_embedding_pmp(ui_input_pmp)
            item_input_pmp = ui_input_pmp.transpose(2,1)
            
            #meta-path attention
            self.item_embedding_mp = Embedding_item(ui_input_mp.shape[1], self.embedding_dim)
            item_embeds_mp = self.item_embedding_mp(item_input_mp)
            user_item_embed_mp = torch.cat((user_embeds_mp, item_embeds_mp), dim=1)
            
            self.attentionLayer_mp = AttentionLayer(self.embedding_dim, item_embeds_mp.shape[1], user_item_embed_mp.shape[1], user_embeds_mp.shape[1], self.drop_ratio)
            at_mp = self.attentionLayer_mp(user_item_embed_mp)
            user_embeds_mp_at = torch.matmul(at_mp, item_embeds_mp)
            
            #dependency meta-path attention
            self.item_embedding_pmp = Embedding_item(ui_input_pmp.shape[1], self.embedding_dim)
            item_embeds_pmp = self.item_embedding_pmp(item_input_pmp)
            user_item_embed_pmp = torch.cat((user_embeds_pmp, item_embeds_pmp), dim=1)
            
            self.attentionLayer_pmp = AttentionLayer(self.embedding_dim, item_embeds_pmp.shape[1], user_item_embed_pmp.shape[1], user_embeds_pmp.shape[1], self.drop_ratio)
            at_pmp = self.attentionLayer_pmp(user_item_embed_pmp)
            user_embeds_pmp_at = torch.matmul(at_pmp, item_embeds_pmp)
            
            #concat
            user_embeddings_mp = torch.cat((user_embeds_mp, user_embeds_mp_at), dim=2)
            user_embeddings_mp = self.MLP(user_embeddings_mp)
            user_embeddings_pmp = torch.cat((user_embeds_pmp, user_embeds_pmp_at), dim=2)
            user_embeddings_pmp = self.MLP(user_embeddings_pmp)
            #fusion
            fusion_rate = torch.sigmoid(torch.add(user_embeddings_mp, user_embeddings_pmp))
            user_embeds_fusion = torch.add(torch.mul(fusion_rate, user_embeddings_mp), torch.mul((1-fusion_rate), user_embeddings_pmp))
            
            #item
            item_embeddings = torch.cat((item_embeds_mp, item_embeds_pmp), dim=2)
            item_embeddings = self.MLP(item_embeddings) 
            return user_embeds_fusion, item_embeddings
    
    def pre_train_forward(self, ui_input_mp, ui_input_pmp, type_info = 'user'):
        user_embeds_fusion, _ = self.paae_fusion(ui_input_mp, ui_input_pmp, type_info)
        logits = self.user_predictor(user_embeds_fusion)
        return logits, user_embeds_fusion
    
    def forward(self, ui_input_mp, ui_input_pmp, type_info = 'user'):
        user_embeds_fusion, item_embeddings = self.paae_fusion(ui_input_mp, ui_input_pmp, type_info)
        user_embeds = torch.tanh(self.transform_layer(user_embeds_fusion))
        return user_embeds, item_embeddings

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, item_dim, output_dim, out_dim, drop_ratio):
        super(AttentionLayer, self).__init__()
        self.drop_ratio = drop_ratio
        #ReLU
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, item_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(self.drop_ratio)
        )

        self.attention = nn.Linear(item_dim, item_dim)
        self.drop = nn.Dropout(self.drop_ratio)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)
        
        self.W = nn.Parameter(torch.empty(size=(out_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, inputs):
        h = torch.tanh(self.mlp(inputs))
        attention_out = torch.tanh(self.attention(h))
        attention_out = torch.matmul(self.W, attention_out)
        weight = torch.softmax(attention_out, dim=1)
        return weight