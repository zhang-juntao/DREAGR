import torch
import numpy as np
import ir_measures
from ir_measures import *

def ndcg_binary_at_k_batch_torch(X_pred, heldout_batch, topK):
    _, idx_topk_pred = torch.topk(X_pred, topK, dim=1, sorted=True)
    max_r, pred_data = getLabel(heldout_batch, idx_topk_pred, topK)
    
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, topK + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, topK + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    
    return torch.Tensor(ndcg)

def recall_at_k_batch_torch(X_pred, heldout_batch, topK):
    """
    Recall@k for predictions [B, I] and ground-truth [B, I].
    """
    batch_users = X_pred.shape[0]
    _, topk_indices = torch.topk(X_pred, topK, dim=1, sorted=False)  # [B, K]
    X_pred_binary = torch.zeros_like(X_pred)
    if torch.cuda.is_available():
        X_pred_binary = X_pred_binary.cuda()
    X_pred_binary[torch.arange(batch_users).unsqueeze(1), topk_indices] = 1
    X_true_binary = (heldout_batch > 0).float()  # .toarray() #  [B, I]
    k_tensor = torch.tensor([topK], dtype=torch.float32)
    if torch.cuda.is_available():
        X_true_binary = X_true_binary.cuda()
        k_tensor = k_tensor.cuda()
    tmp = (X_true_binary * X_pred_binary).sum(dim=1).float()
    recall = tmp / torch.min(k_tensor, X_true_binary.sum(dim=1).float())
    return recall


def HR__binary_at_k_batch_torch(X_pred, heldout_batch, topK):
    _, idx_topk_pred = torch.topk(X_pred, topK, dim=1, sorted=True)
    max_r, pred_data = getLabel(heldout_batch, idx_topk_pred, topK)
    
    pred_data_sum = pred_data.sum(axis=1)
    max_r_sum = max_r.sum(axis=1)
    
    hr = pred_data_sum / max_r_sum
    hr[np.isnan(hr)] = 0
    return torch.Tensor(hr)


def getLabel(test_data, pred_data, topK):
    test_data_temp = test_data.tolist()
    test_data_list = []
    for i in range(len(test_data_temp)):
        temp = [j for j in range(len(test_data_temp[i])) if test_data_temp[i][j] > 0.0]
        test_data_list.append(temp)
    
    r = []
    for i in range(len(test_data_list)):
        t0 = test_data_list[i]
        t1 = pred_data[i].tolist()
        tt0 = list(set(t0) & set(t1))
        r.append(tt0)
    
    pred_data_result = []
    for i in range(len(pred_data)):
        temp = []
        t1 = pred_data[i].tolist()
        
        for j in range(len(t1)):
            if t1[j] in r[i]:
                temp.append(1)
            else:
                temp.append(0)
        pred_data_result.append(temp)
        
    groundTrue = []
    for i in range(len(test_data_list)):
        temp = []
        if len(test_data_list[i]) >= topK:
            temp.extend([1] * topK)
        else:
            temp.extend([1] * len(test_data_list[i]))
            temp.extend([0] * (topK - len(test_data_list[i])))
            
        groundTrue.append(temp)
    
    
    return np.array(groundTrue).astype('float'), np.array(pred_data_result).astype('float')
    
