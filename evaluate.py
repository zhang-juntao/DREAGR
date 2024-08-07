import torch
import numpy as np
import metrics
import gc

def evaluate_user(model, eval_loader, mode='pretrain', topK_list = list):
    """ evaluate model on recommending items to users """
    ndcgs = []
    recalls = []
    hrs = []
    
    for topK in topK_list:
        model.eval()
        eval_loss = 0.0
        ndcg_list, recall_list, hr_list = [], [], []
        eval_preds = []
        with torch.no_grad():
            for batch_index, eval_data in enumerate(eval_loader):
                (users, fold_in_items_mp, fold_in_items_pmp, held_out_items) = eval_data
                if mode == 'pretrain':
                    recon_batch, emb = model.paee.pre_train_forward(fold_in_items_mp, fold_in_items_pmp, type_info = 'user')
                else:
                    recon_batch = model.group_predictor(model.paee(fold_in_items_mp, fold_in_items_pmp))
    
                loss = model.multinomial_loss(recon_batch, held_out_items)
                eval_loss += loss.item()
                fold_in_items = torch.add(fold_in_items_mp, fold_in_items_pmp)
                #fold_in_items[fold_in_items > 1] = 1
                fold_in_items = fold_in_items.cpu().numpy()
                recon_batch = torch.softmax(recon_batch, 1)  # softmax over the item set to get normalized scores.
                recon_batch[fold_in_items.nonzero()] = -np.inf
    
                ndcg = metrics.ndcg_binary_at_k_batch_torch(recon_batch, held_out_items, topK)
                recall = metrics.recall_at_k_batch_torch(recon_batch, held_out_items, topK)
                hr = metrics.HR__binary_at_k_batch_torch(recon_batch, held_out_items, topK)
    
                ndcg_list.append(ndcg)
                recall_list.append(recall)
                hr_list.append(hr)
    
                eval_preds.append(recon_batch.cpu().numpy())
                del users, fold_in_items, held_out_items, recon_batch, fold_in_items_mp, fold_in_items_pmp
        gc.collect()
        eval_loss = eval_loss / len(eval_loader)
        ndcg_temp = torch.mean(torch.cat(ndcg_list), dtype=torch.float32)
        recall_temp = torch.mean(torch.cat(recall_list), dtype=torch.float32)
        hr_temp = torch.mean(torch.cat(hr_list), dtype=torch.float32)
    
        ndcgs.append(round(ndcg_temp.item(), 4))
        recalls.append(round(recall_temp.item(), 4))
        hrs.append(round(hr_temp.item(), 4))
    return eval_loss, ndcgs, recalls, hrs


def evaluate_group(model, eval_group_loader, type_info = 'test', topK_list = list):
    """ evaluate model on recommending items to groups """
    ndcgs = []
    recalls = []
    hrs = []
    
    for topK in topK_list:
        model.eval()
        eval_loss = 0.0
        ndcg_list, recall_list, hr_list = [], [], []
        eval_preds = []
    
        with torch.no_grad():
            for batch_idx, data in enumerate(eval_group_loader):
                group, mp_info, pmp_info = data
                _, group_mask_mp, group_items_mp, user_items_mp = mp_info
                _, group_mask_pmp, group_items_pmp, user_items_pmp = pmp_info
                
                recon_batch, _ = model(group, mp_info, pmp_info, type_info = 'test')
    
                loss = model.loss(recon_batch, (group_items_mp+group_items_pmp), (user_items_mp+user_items_pmp), (group_mask_mp +group_mask_pmp))
                eval_loss += loss.item()
                result = recon_batch.softmax(1)
                heldout_data = (group_items_mp+group_items_pmp)
                heldout_data[heldout_data > 1] = 1
                
                ndcg = metrics.ndcg_binary_at_k_batch_torch(result, heldout_data, topK)
                recall = metrics.recall_at_k_batch_torch(result, heldout_data, topK)
                hr = metrics.HR__binary_at_k_batch_torch(result, heldout_data, topK)
                
                ndcg_list.append(ndcg)
                recall_list.append(recall)
                hr_list.append(hr)
    
                eval_preds.append(recon_batch.cpu().numpy())
                del group, mp_info, pmp_info
        gc.collect()
        eval_loss = eval_loss / len(eval_group_loader)
        ndcg_temp = torch.mean(torch.cat(ndcg_list), dtype=torch.float32)
        recall_temp = torch.mean(torch.cat(recall_list), dtype=torch.float32)
        hr_temp = torch.mean(torch.cat(hr_list), dtype=torch.float32)
    
        ndcgs.append(round(ndcg_temp.item(), 4))
        recalls.append(round(recall_temp.item(), 4))
        hrs.append(round(hr_temp.item(), 4))
    
    return eval_loss, ndcgs, recalls, hrs

