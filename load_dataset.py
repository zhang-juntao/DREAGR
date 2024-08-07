import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from torch.utils import data

#load user train dataset
class LoadTrainUserdataset(data.Dataset):
    def __init__(self, dataset, user_path):
        self.dataset = dataset
        self.user_data_dir = user_path
        self.train_data_ui_mp, self.train_data_ui_pmp = self._load_train_data()
        self.user_list = list(range(self.num_users))
        
    def __len__(self):
        return len(self.user_list)
    
    def __getitem__(self, index):
        """ load user_id, binary vector over items """
        user = self.user_list[index]
        user_items_mp = torch.from_numpy(self.train_data_ui_mp[user, :].toarray()).squeeze()
        user_items_pmp = torch.from_numpy(self.train_data_ui_pmp[user, :].toarray()).squeeze()
        
        return torch.from_numpy(np.array([user], dtype=np.int32)), user_items_mp, user_items_pmp
    
    def _load_train_data(self):
        """ load training user-item interactions as a sparse matrix """
        datainfo_list = ['Train_mp.csv', 'Train_pmp.csv', 'Val_mp_tr.csv', 'Val_pmp_tr.csv', 'Test_mp_tr.csv', 'Test_pmp_tr.csv']
        df = pd.DataFrame()
        for path in datainfo_list:
            pathfile  = self.user_data_dir + path
            temp = pd.read_csv(pathfile)
            df = df.append(temp)
        df.index = [i for i in range(len(df))]
        
        n_users, self.num_items = int(df['user'].max() + 1), int(df['item'].max() + 1)
        print("# users", n_users, "# items", self.num_items)
        
        #load meta-path training data
        train_ui_mp = pd.read_csv(self.user_data_dir + datainfo_list[0])
        rows_ui_mp, cols_ui_mp = train_ui_mp['user'], train_ui_mp['item']
        #load dependency meta-path training data
        train_ui_pmp = pd.read_csv(self.user_data_dir + datainfo_list[1])
        rows_ui_pmp, cols_ui_pmp = train_ui_pmp['user'], train_ui_pmp['item']
        
        start_idx = min(train_ui_mp['user'].min(), train_ui_pmp['user'].min())
        end_idx = max(train_ui_mp['user'].max(), train_ui_pmp['user'].max())
        self.num_users = end_idx - start_idx + 1
        
        data_ui_mp = sp.csr_matrix((np.ones_like(rows_ui_mp), (rows_ui_mp, cols_ui_mp)), dtype='float32',
                                shape=(end_idx - start_idx + 1, self.num_items))
        
        data_ui_pmp = sp.csr_matrix((np.ones_like(rows_ui_pmp), (rows_ui_pmp, cols_ui_pmp)), dtype='float32',
                                shape=(end_idx - start_idx + 1, self.num_items))

        return data_ui_mp, data_ui_pmp


#load user test dataset
class LoadTestUserdataset(data.Dataset):
    def __init__(self, dataset, num_items, user_path, datatype='Val'):
        self.dataset = dataset
        self.num_items = num_items
        self.user_data_dir = user_path
        self.ui_data_tr_mp, self.ui_data_tr_pmp, self.ui_data_te = self._load_test_data(datatype)
        
    def __len__(self):
        return len(self.user_list)
    
    def __getitem__(self, index):
        """ load user_id, user-items """
        user = self.user_list[index]
        fold_in_mp, fold_in_pmp = self.ui_data_tr_mp[user, :].toarray(), self.ui_data_tr_pmp[user, :].toarray()  # [I], [I]
        held_out = self.ui_data_te[user, :].toarray()
        
        return user, torch.from_numpy(fold_in_mp).squeeze(), torch.from_numpy(fold_in_pmp).squeeze(), held_out.squeeze()  # user, fold-in items, fold-out items.
    
    def _load_test_data(self, datatype='Val'):
        """ load user-item interactions of test user sets """
        if datatype == 'Val':
            ui_tr_mp = self.user_data_dir + 'Val_mp_tr.csv'
            ui_tr_pmp = self.user_data_dir + 'Val_pmp_tr.csv'
            ui_te_path = self.user_data_dir + 'Val_te.csv'
        elif datatype == 'Test':
            ui_tr_mp = self.user_data_dir + 'Test_mp_tr.csv'
            ui_tr_pmp = self.user_data_dir + 'Test_pmp_tr.csv'
            ui_te_path = self.user_data_dir + 'Test_te.csv'
        
        ui_tr_mp_df, ui_tr_pmp_df = pd.read_csv(ui_tr_mp), pd.read_csv(ui_tr_pmp)
        ui_te_df = pd.read_csv(ui_te_path)
        
        start_idx = min(ui_tr_mp_df['user'].min(), ui_tr_pmp_df['user'].min(), ui_te_df['user'].min())
        end_idx= max(ui_tr_mp_df['user'].max(), ui_tr_pmp_df['user'].max(), ui_te_df['user'].max())
        
        rows_tr_mp, cols_tr_mp = ui_tr_mp_df['user'] - start_idx, ui_tr_mp_df['item']
        rows_tr_pmp, cols_tr_pmp = ui_tr_pmp_df['user'] - start_idx, ui_tr_pmp_df['item']
        
        rows_te, cols_te = ui_te_df['user'] - start_idx, ui_te_df['item']
        self.user_list = list(range(0, end_idx - start_idx + 1))
    
        ui_data_tr_mp = sp.csr_matrix((np.ones_like(rows_tr_mp), (rows_tr_mp, cols_tr_mp)), dtype='float32',
                                   shape=(end_idx - start_idx + 1, self.num_items))
        ui_data_tr_pmp = sp.csr_matrix((np.ones_like(rows_tr_pmp), (rows_tr_pmp, cols_tr_pmp)), dtype='float32',
                                   shape=(end_idx - start_idx + 1, self.num_items))
        
        ui_data_te = sp.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), dtype='float32',
                                   shape=(end_idx - start_idx + 1, self.num_items))
        return ui_data_tr_mp, ui_data_tr_pmp, ui_data_te

#load group train data
class LoadTrainGroupdataset(data.Dataset):
    def __init__(self, dataset, user_path, group_path, num_items, num_negatives):
        self.dataset = dataset
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_data_dir = user_path
        self.group_data_dir = group_path
        self.user_data = self._load_user_data()
        self.group_data_mp, self.group_data_pmp, self.group_users_mp, self.group_users_pmp = self._load_group_data()
        self.group_inputs_mp = [self.user_data[self.group_users_mp[g]] for g in self.groups_list]
        self.group_inputs_pmp = [self.user_data[self.group_users_pmp[g]] for g in self.groups_list]
    
    def __len__(self):
        return len(self.groups_list)
    
    def get_corrupted_users(self, group, ptype = 'mp'):
        p = np.ones(self.num_users)
        if ptype == 'mp':
            p[self.group_users_mp[group]] = 0
            p = normalize([p], norm='l1')[0]
            item_biased = normalize(self.user_data[:, self.group_data_mp[group].indices].sum(1).squeeze(), norm='l1')[0]
            p = item_biased + p
            negative_users = torch.multinomial(torch.from_numpy(p), self.num_negatives)
        elif ptype == 'pmp':
            p[self.group_users_pmp[group]] = 0
            p = normalize([p], norm='l1')[0]
            item_biased = normalize(self.user_data[:, self.group_data_pmp[group].indices].sum(1).squeeze(), norm='l1')[0]
            p = item_biased + p
            negative_users = torch.multinomial(torch.from_numpy(p), self.num_negatives)
        return negative_users

    
    def __getitem__(self, index):
        group = self.groups_list[index]
        #meta-path
        user_ids_mp = torch.from_numpy(np.array(self.group_users_mp[group], np.int32))
        group_items_mp = torch.from_numpy(self.group_data_mp[group].toarray().squeeze())

        corrupted_group_mp = self.get_corrupted_users(group, ptype = 'mp')
        corrupted_user_items_mp = torch.from_numpy(self.user_data[corrupted_group_mp].toarray().squeeze())
        
        group_length_mp = self.max_group_size - list(user_ids_mp).count(self.padding_idx)
        group_mask_mp = torch.from_numpy(np.concatenate([np.zeros(group_length_mp, dtype=np.float32), (-1) * np.inf *
                                                      np.ones(self.max_group_size - group_length_mp,
                                                              dtype=np.float32)]))

        user_items_mp = torch.from_numpy(self.group_inputs_mp[group].toarray())
        
        mp_info = [user_ids_mp, group_mask_mp, group_items_mp, user_items_mp, corrupted_user_items_mp]
        
        #dependency meta-path
        user_ids_pmp = torch.from_numpy(np.array(self.group_users_pmp[group], np.int32))
        group_items_pmp = torch.from_numpy(self.group_data_pmp[group].toarray().squeeze())

        corrupted_group_pmp = self.get_corrupted_users(group, ptype = 'pmp')
        corrupted_user_items_pmp = torch.from_numpy(self.user_data[corrupted_group_pmp].toarray().squeeze())

        # group mask to create fixed-size padded groups.
        group_length_pmp = self.max_group_size - list(user_ids_pmp).count(self.padding_idx)
        group_mask_pmp = torch.from_numpy(np.concatenate([np.zeros(group_length_pmp, dtype=np.float32), (-1) * np.inf *
                                                      np.ones(self.max_group_size - group_length_pmp,
                                                              dtype=np.float32)]))

        user_items_pmp = torch.from_numpy(self.group_inputs_pmp[group].toarray())
        
        pmp_info = [user_ids_pmp, group_mask_pmp, group_items_pmp, user_items_pmp, corrupted_user_items_pmp]
        
        return torch.tensor([group]), mp_info, pmp_info
    
    def _load_group_data(self):
        """ load training group-item interactions as a sparse matrix and user-group memberships """
        #元路径
        path_gi_mp = self.group_data_dir + 'Train_mp.csv'
        df_gi_mp = pd.read_csv(path_gi_mp)
        
        start_idx, end_idx = df_gi_mp['group'].min(), df_gi_mp['group'].max()
        self.n_groups = end_idx - start_idx + 1
        rows_gi_mp, cols_gi_mp = df_gi_mp['group'] - start_idx, df_gi_mp['item']
        data_gi_mp = sp.csr_matrix((np.ones_like(rows_gi_mp), (rows_gi_mp, cols_gi_mp)), dtype='float32',
                                shape=(self.n_groups, self.num_items))
        
        #先决元路径
        path_gi_pmp = self.group_data_dir + 'Train_pmp.csv'
        df_gi_pmp = pd.read_csv(path_gi_pmp)
        start_idx_pmp, end_idx_pmp = df_gi_pmp['group'].min(), df_gi_pmp['group'].max()
        assert self.n_groups == end_idx_pmp - start_idx_pmp + 1
        rows_gi_pmp, cols_gi_pmp = df_gi_pmp['group'] - start_idx_pmp, df_gi_pmp['item']
        data_gi_pmp = sp.csr_matrix((np.ones_like(rows_gi_pmp), (rows_gi_pmp, cols_gi_pmp)), dtype='float32',
                                shape=(self.n_groups, self.num_items))
        
        #meta-path
        path_ug_mp = 'data/'+self.dataset+'/groupMember_mp.csv'
        df_ug_mp = pd.read_csv(path_ug_mp).astype(int)
        df_ug_mp.columns = ['group', 'user']
        df_ug_train_mp = df_ug_mp[df_ug_mp.group.isin(range(start_idx, end_idx + 1))]
        df_ug_train_mp = df_ug_train_mp.sort_values('group')
        
        ##dependency meta-path
        path_ug_pmp = 'data/'+self.dataset+'/groupMember_pmp.csv'
        df_ug_pmp = pd.read_csv(path_ug_pmp).astype(int)
        df_ug_pmp.columns = ['group', 'user']
        df_ug_train_pmp = df_ug_pmp[df_ug_pmp.group.isin(range(start_idx_pmp, end_idx_pmp + 1))]
        df_ug_train_pmp = df_ug_train_pmp.sort_values('group')
        
        self.max_group_size = max(df_ug_train_mp.groupby('group').size().max(), df_ug_train_pmp.groupby('group').size().max())
        
        #meta-path
        gu_list_train_mp = df_ug_train_mp.groupby('group')['user'].apply(list).reset_index()
        gu_list_train_mp['user'] = list(map(lambda x: x + [self.padding_idx-1] * (self.max_group_size - len(x)),
                                          gu_list_train_mp.user))
        data_gu_mp = np.squeeze(np.array(gu_list_train_mp[['user']].values.tolist()))
        self.groups_list = list(range(0, end_idx - start_idx + 1))
        assert len(df_ug_train_mp['group'].unique()) == self.n_groups
        
        ##dependency meta-path
        gu_list_train_pmp = df_ug_train_pmp.groupby('group')['user'].apply(list).reset_index()
        gu_list_train_pmp['user'] = list(map(lambda x: x + [self.padding_idx-1] * (self.max_group_size - len(x)),
                                          gu_list_train_pmp.user))
        data_gu_pmp = np.squeeze(np.array(gu_list_train_pmp[['user']].values.tolist()))
        assert len(df_ug_train_pmp['group'].unique()) == self.n_groups
        
        print("# training groups: {}, # max train group size: {}".format(self.n_groups, self.max_group_size))
        return data_gi_mp, data_gi_pmp, data_gu_mp, data_gu_pmp
    
    
    def _load_user_data(self):
        """ load user-item interactions of all users that appear in training groups, as a sparse matrix """
        #include Train and TrainPre
        datainfo_list = ['Train_mp.csv', 'Train_pmp.csv', 'Val_mp_tr.csv', 'Val_pmp_tr.csv', 'Test_mp_tr.csv', 'Test_pmp_tr.csv']
        df_ui = pd.DataFrame()
        for path in datainfo_list:
            pathfile  = self.user_data_dir + path
            temp = pd.read_csv(pathfile)
            df_ui = df_ui.append(temp)

        self.num_users = df_ui['user'].max() + 1
        self.padding_idx = self.num_users
        assert self.num_items == df_ui['item'].max() + 1
        rows_ui, cols_ui = df_ui['user'], df_ui['item']

        data_ui = sp.csr_matrix((np.ones_like(rows_ui), (rows_ui, cols_ui)), dtype='float32',
                                shape=(self.num_users, self.num_items))
        return data_ui

#load group train data
class LoadTestgroupdataset(data.Dataset):
    def __init__(self, dataset, user_path, group_path, num_items, padding_idx, datatype='Val'):
        self.dataset = dataset
        self.num_items = num_items
        self.padding_idx = padding_idx
        self.user_data_dir = user_path
        self.group_data_dir = group_path
        
        self.groups_list = []
        self.user_data = self._load_user_data()
        self.data_gi_mp, self.data_gi_pmp, self.data_gu_mp, self.data_gu_pmp = self._load_group_data(datatype)
        
    def __len__(self):
        return len(self.groups_list)

    def __getitem__(self, index):
        """ load group_id, padded group users, mask, group items, group member items """
        group = self.groups_list[index]
        #meta-path
        user_ids_mp = self.data_gu_mp[group]
        length_mp = self.max_gsize - list(user_ids_mp).count(self.padding_idx)
        mask_mp = torch.from_numpy(np.concatenate([np.zeros(length_mp, dtype=np.float32), (-1) * np.inf *
                                                np.ones(self.max_gsize - length_mp, dtype=np.float32)]))
        group_items_mp = torch.from_numpy(self.data_gi_mp[group].toarray().squeeze())
        user_items_mp = torch.from_numpy(self.user_data[user_ids_mp].toarray().squeeze())
        mp_info = [torch.tensor(user_ids_mp), mask_mp, group_items_mp, user_items_mp]
        
        ##dependency meta-path
        user_ids_pmp = self.data_gu_pmp[group]
        length_pmp = self.max_gsize - list(user_ids_pmp).count(self.padding_idx)
        mask_pmp = torch.from_numpy(np.concatenate([np.zeros(length_pmp, dtype=np.float32), (-1) * np.inf *
                                                np.ones(self.max_gsize - length_pmp, dtype=np.float32)]))  # [G]
        group_items_pmp = torch.from_numpy(self.data_gi_pmp[group].toarray().squeeze())
        user_items_pmp = torch.from_numpy(self.user_data[user_ids_pmp].toarray().squeeze())
        pmp_info = [torch.tensor(user_ids_pmp), mask_pmp, group_items_pmp, user_items_pmp]
        
        return torch.tensor([group]), mp_info, pmp_info

    def _load_user_data(self):
        """ load user-item interactions of all users that appear in training groups, as a sparse matrix """
        datainfo_list = ['Train_mp.csv', 'Train_pmp.csv', 'Val_mp_tr.csv', 'Val_pmp_tr.csv', 'Test_mp_tr.csv', 'Test_pmp_tr.csv']
        df_ui = pd.DataFrame()
        for path in datainfo_list:
            pathfile  = self.user_data_dir + path
            temp = pd.read_csv(pathfile)
            df_ui = df_ui.append(temp)

        self.num_users = df_ui['user'].max() + 1
        self.padding_idx = self.num_users
        assert self.num_items == df_ui['item'].max() + 1
        rows_ui, cols_ui = df_ui['user'], df_ui['item']

        data_ui = sp.csr_matrix((np.ones_like(rows_ui), (rows_ui, cols_ui)), dtype='float32',
                                shape=(self.num_users, self.num_items))
        return data_ui

    def _load_group_data(self, datatype='Val'):
        """ load test group-item interactions as a sparse matrix and user-group memberships """
        if datatype=='Val':
            path_gi_mp = self.group_data_dir + 'Val_mp.csv'
            path_gi_pmp = self.group_data_dir + 'Val_pmp.csv'

        elif datatype=='Test':
            path_gi_mp = self.group_data_dir + 'Test_mp.csv'
            path_gi_pmp = self.group_data_dir + 'Test_pmp.csv'
            
        df_gi_mp = pd.read_csv(path_gi_mp)
        df_gi_pmp = pd.read_csv(path_gi_pmp)
        start_idx = min(df_gi_mp['group'].min(), df_gi_pmp['group'].min())
        end_idx = max(df_gi_mp['group'].max(), df_gi_pmp['group'].max())
        
        #meta-path
        self.n_groups = end_idx - start_idx + 1
        rows_gi, cols_gi = df_gi_mp['group'] - start_idx, df_gi_mp['item']
        data_gi_mp = sp.csr_matrix((np.ones_like(rows_gi), (rows_gi, cols_gi)), dtype='float32',
                                shape=(self.n_groups, self.num_items))
        
        ##dependency meta-path
        rows_gi_pmp, cols_gi_pmp = df_gi_pmp['group'] - start_idx, df_gi_pmp['item']
        data_gi_pmp = sp.csr_matrix((np.ones_like(rows_gi_pmp), (rows_gi_pmp, cols_gi_pmp)), dtype='float32',
                                shape=(self.n_groups, self.num_items))
        
        path_ug_mp = 'data/'+self.dataset+'/groupMember_mp.csv'
        df_ug_mp = pd.read_csv(path_ug_mp).astype(int)
        df_ug_mp.columns = ['group', 'user']
        df_ug_test_mp = df_ug_mp[df_ug_mp.group.isin(range(start_idx, end_idx + 1))]
        df_ug_test_mp = df_ug_test_mp.sort_values('group')
        
        ##dependency meta-path
        path_ug_pmp = 'data/'+self.dataset+'/groupMember_pmp.csv'
        df_ug_pmp = pd.read_csv(path_ug_pmp).astype(int)
        df_ug_pmp.columns = ['group', 'user']
        df_ug_test_pmp = df_ug_pmp[df_ug_pmp.group.isin(range(start_idx, end_idx + 1))]
        df_ug_test_pmp = df_ug_test_pmp.sort_values('group')
        
        self.max_gsize = max(df_ug_test_mp.groupby('group').size().max(), df_ug_test_pmp.groupby('group').size().max())
        
        gu_list_test_mp = df_ug_test_mp.groupby('group')['user'].apply(list).reset_index()
        gu_list_test_mp['user'] = list(map(lambda x: x + [self.padding_idx-1] * (self.max_gsize - len(x)),
                                          gu_list_test_mp.user))
        data_gu_mp = np.squeeze(np.array(gu_list_test_mp[['user']].values.tolist()))
        assert len(df_ug_test_mp['group'].unique()) == self.n_groups
        
        gu_list_test_pmp = df_ug_test_pmp.groupby('group')['user'].apply(list).reset_index()
        gu_list_test_pmp['user'] = list(map(lambda x: x + [self.padding_idx-1] * (self.max_gsize - len(x)),
                                          gu_list_test_pmp.user))
        data_gu_pmp = np.squeeze(np.array(gu_list_test_pmp[['user']].values.tolist()))
        assert len(df_ug_test_pmp['group'].unique()) == self.n_groups
        self.groups_list = list(range(0, end_idx - start_idx + 1))
        
        return data_gi_mp, data_gi_pmp, data_gu_mp, data_gu_pmp
        