# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:05:55 2023

@author: junta
"""

import torch
import time
import gc
import argparse
import numpy as np
from load_dataset import LoadTrainUserdataset, LoadTestUserdataset, LoadTrainGroupdataset, LoadTestgroupdataset
from torch.utils.data import DataLoader
from model import DREAGR
from evaluate import evaluate_group, evaluate_user
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser('DEAGR model')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='MOOCCube')
args = parser.parse_args()
parser.add_argument('--user_path', type=str, default='./data/'+args.dataset+'/userRating')
parser.add_argument('--group_path', type=str, default='./data/'+args.dataset+'/groupRating')

parser.add_argument('--topK', type=list, default = [5, 10, 20], help= 'top K')

parser.add_argument('--lr', type=float, default = 0.001, help= 'Learning rate')
parser.add_argument('--num_negatives', type=int, default = 4, help= '负采样')
parser.add_argument('--embedding_dim', type=int, default = 64, help= 'Embedding size')

parser.add_argument('--epochs', type=int, default = 50, help= '运行次数')
parser.add_argument('--batch_size', type=int, default = 64, help= 'batch_size')
parser.add_argument('--wd', type=float, default = 0.0, help= 'weight_decay')
parser.add_argument('--drop_ratio', type=float, default = 0.0, help= 'drop_ratio')

parser.add_argument('--aggregator', type=str, default='attention', help='choice of group preference aggregator',
                    choices=['meanpool', 'attention'])

parser.add_argument('--pretrain_epochs', type=int, default=10, help='# pre-train epochs for user encoder layer')
parser.add_argument('--save', type=str, default='save_model/model_user.pt', help='path to save the final model')
parser.add_argument('--save_group', type=str, default='save_model/model_group.pt', help='path to save the final model')
args = parser.parse_args()

# Load data
train_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': True}
eval_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': True}

#load dataset
user_train_dataset = LoadTrainUserdataset(args.dataset, args.user_path)
num_items, num_users = user_train_dataset.num_items, user_train_dataset.num_users
user_val_dataset = LoadTestUserdataset(args.dataset, num_items, args.user_path, datatype='Val')
user_test_dataset = LoadTestUserdataset(args.dataset, num_items, args.user_path, datatype='Test')

# Define data loaders on user interactions.
train_loader = DataLoader(user_train_dataset, **train_params)
val_loader = DataLoader(user_val_dataset, **train_params)
test_loader = DataLoader(user_test_dataset, **eval_params)

group_train_dataset = LoadTrainGroupdataset(args.dataset, args.user_path, args.group_path, num_items, args.num_negatives)
padding_idx = group_train_dataset.padding_idx
group_val_dataset = LoadTestgroupdataset(args.dataset, args.user_path, args.group_path, num_items, padding_idx, datatype='Val')
group_test_dataset = LoadTestgroupdataset(args.dataset, args.user_path, args.group_path, num_items, padding_idx, datatype='Test')

# # Define data loaders on group interactions.
group_train_loader = DataLoader(group_train_dataset, **train_params)
group_val_loader = DataLoader(group_val_dataset, **train_params)
group_test_loader = DataLoader(group_test_dataset, **train_params)


ndcg_list, hr_list = [], []
#build model
dreagr = DREAGR(num_users, num_items, args.embedding_dim, drop_ratio = args.drop_ratio, aggregator_type = args.aggregator)

best_user_ndcg, best_group_ndcg = -np.inf, -np.inf
optimizer_ur = torch.optim.Adam(dreagr.parameters(), lr=0.01, weight_decay=0.001)
print("Pre-training model on user-item interactions")
for epoch in range(0, args.pretrain_epochs):
    epoch_start_time = time.time()
    dreagr.train()
    
    train_user_loss = 0.0
    start_time = time.time()
    
    for batch_index, data in enumerate(train_loader):
        optimizer_ur.zero_grad()
        (train_users, train_items_mp, train_items_pmp) = data
        user_logits, user_embeds = dreagr.paee.pre_train_forward(train_items_mp, train_items_pmp, type_info = 'user')
        user_loss_mp = dreagr.user_loss(user_logits, train_items_mp)
        user_loss_pmp = dreagr.user_loss(user_logits, train_items_pmp)
        
        user_loss = user_loss_mp + user_loss_pmp
        user_loss.backward()
        train_user_loss += user_loss.item()
        optimizer_ur.step()
        del train_users, train_items_mp, train_items_pmp, user_logits, user_embeds
    elapsed = time.time() - start_time
    
    train_user_loss = train_user_loss / len(train_loader)
    test_loss, ndcg, _, hr = evaluate_user(dreagr, val_loader, mode='pretrain', topK_list = args.topK)
    print('|user_epoch {:3d} |  time {:4.2f} | loss {:4.2f}| hr {:4.4f} '.format(epoch + 1, elapsed, train_user_loss, hr[0]))
    if hr[0] > best_user_ndcg:
        torch.save(dreagr.state_dict(), args.save)
        best_user_ndcg = hr[0]

print("Load best pre-trained user embedding")
dreagr.load_state_dict(torch.load(args.save))    
test_loss, ndcg, _, hr = evaluate_user(dreagr, test_loader, mode='pretrain', topK_list = args.topK)
print(f"[Epoch {epoch + 1}] User, Hit@{args.topK}: {hr}, NDCG@{args.topK}: {ndcg}")

print("Initializing group recommender with pre-train user embedding")
dreagr.group_predictor.weight.data = dreagr.paee.user_predictor.weight.data
optimizer_gr = torch.optim.Adam(dreagr.parameters(), lr=args.lr, weight_decay=args.wd)

for epoch in range(0, args.epochs):
    start_time = time.time()
    dreagr.train()
    train_group_epoch_loss = 0.0
    for batch_index, data in enumerate(group_train_loader):
        group, mp_info, pmp_info = data
        _, group_mask_mp, group_items_mp, user_items_mp, _ = mp_info
        _, group_mask_pmp, group_items_pmp, user_items_pmp, _ = pmp_info
        
        optimizer_gr.zero_grad()
        dreagr.zero_grad()
        group_logits, group_embeds = dreagr(group.squeeze(), mp_info, pmp_info, type_info = 'train')
        group_loss = dreagr.loss(group_logits, (group_items_mp+group_items_pmp), (user_items_mp+user_items_pmp), (group_mask_mp +group_mask_pmp))
        group_loss.backward()
        train_group_epoch_loss += group_loss.item()
        optimizer_gr.step()
        #删除变量
        del group, mp_info, pmp_info, group_items_mp, user_items_mp, \
        group_items_pmp, user_items_pmp, group_mask_mp, group_mask_pmp
    gc.collect()
    elapsed = time.time() - start_time
    #if epoch % 5 == 0:
    val_loss_group, ndcgs_val_group, _, hrs_val_group = evaluate_group(dreagr, group_val_loader, type_info = 'test', topK_list = args.topK)
    print("| epoch {:3d} | elapsed: {} | Val loss: {}".format((epoch+1), round(elapsed, 4), round(val_loss_group, 4)), 
          f"Hit@{args.topK}: {hrs_val_group}, NDCG@{args.topK}: {ndcgs_val_group}")
    if hrs_val_group[2] > best_group_ndcg:
        with open(args.save_group, 'wb') as f:
            torch.save(dreagr, f)
        best_group_ndcg = hrs_val_group[2]

with open(args.save_group, 'rb') as f:
    deagr = torch.load(f)

test_loss_group, ndcgs_test_group, _, hrs_test_group = evaluate_group(deagr, group_test_loader, type_info = 'test', topK_list = args.topK)
print('| Group test loss {:4.4f} |'.format(test_loss_group), f"Hit@{args.topK}: {hrs_test_group}, NDCG@{args.topK}: {ndcgs_test_group}")


