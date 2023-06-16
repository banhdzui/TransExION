'''
Created on 04 Jun 2022

@author: danhbuithi

'''
import json 
import h5py
import numpy as np
 
import torch 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

from spectrum.io import load_mgf_file, convert_raw2refined_spectra
import math

C_MAX_PEAK_DIFF = 299
C_NUM_DEFFECT_BIN = 100
CLS = 1
       
def mspair_collate_fn(batch):
    '''
    Collate lists of spectrum pairs into batches
    '''
    X = []
    X_mask = []
    Y = []
    Y_mask = []
    labels = []
    
    for mz_f, mz_mask, loss_f, loss_mask, target in batch:
        X.append(mz_f)
        X_mask.append(mz_mask)
        Y.append(loss_f)
        Y_mask.append(loss_mask)
        
        labels.append(target)
        
    X = pad_sequence(X, batch_first=True)
    X_mask = pad_sequence(X_mask, batch_first=True, padding_value=True)
    Y = pad_sequence(Y, batch_first=True)
    Y_mask = pad_sequence(Y_mask, batch_first=True, padding_value=True)
    
    labels = torch.tensor(labels, dtype=torch.double)
    return X, X_mask, Y, Y_mask, labels
    
def vectorize_mass_diff(x, dmass):
    '''
    Convert a mass difference matrix into aligned matrix 
    '''
    n, m = x.shape
    f = torch.zeros(n+1, C_MAX_PEAK_DIFF+1, dtype=torch.long)
    x = np.round(x * C_NUM_DEFFECT_BIN, 0).astype(int)
    
    dmass = np.round(dmass*C_NUM_DEFFECT_BIN, 0).astype(int)
    r = dmass // C_NUM_DEFFECT_BIN
    c = dmass % C_NUM_DEFFECT_BIN
    if r >= C_MAX_PEAK_DIFF:
        f[0, 0] = CLS #CLS token
    else:
        f[0, r+1] = c + 2 
        
    for i in range(n):
        for j in range(m):
            mass_diff = x[i, j]
            r = mass_diff // C_NUM_DEFFECT_BIN
            c = mass_diff % C_NUM_DEFFECT_BIN
            if r >= C_MAX_PEAK_DIFF: continue
            f[i+1, r+1] = c + 2
    return f

def compute_feature_by_sim_matrix(x, y, dmass):
    '''
    Convert a mass difference matrix into aligned matrix and its padding mask 
    '''
    n = x.shape[0]
    mass_diff = np.abs(x.reshape(-1, 1) - y)
    f = vectorize_mass_diff(mass_diff, dmass)
    mask = torch.zeros(n+1, dtype=torch.bool)
    return f, mask

class MSDataset(Dataset):
    '''
    Set of MS/MS spectra which are saved in a h5py format file
    '''
    def __init__(self, db_file_name):
        self.db_file = db_file_name
        
        with h5py.File(self.db_file, 'r') as db:
            self.nsamples =len(db['data'])
            
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, index):
        with h5py.File(self.db_file, 'r') as db:
            i_data = json.loads(db['data'][index])
        return i_data
    
    def get_inchikey_indices(self):
        inchikey_indices = {}
        with h5py.File(self.db_file, 'r') as db:
            for i, x in enumerate(db['data']):
                s = json.loads(x) 
                key = s[1]
                
                if key not in inchikey_indices: 
                    inchikey_indices[key] = []
                    
                inchikey_indices[key].append(i)
        return inchikey_indices
    
class MSMemoryDataset(Dataset):
    '''
    Set of MS/MS spectra which are loaded in memory
    '''
    def __init__(self, db_file_name):
        ms_data = load_mgf_file(db_file_name, use_drug=True)
        self.db = convert_raw2refined_spectra(ms_data)
        
            
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, index):
        return self.db[index]
    
    
class MSPairSet(Dataset):
    '''
    Set of MS/MS spectrum pairs which are specified in the index_file file
    '''
    def __init__(self, source_db, query_db, index_file_name):
        self.source_db = source_db 
        self.query_db = query_db
        
        self.index_file = index_file_name    
        with h5py.File(self.index_file, 'r') as db: 
            self.npairs = len(db['data'])
        
    def __len__(self):
        return self.npairs
    
    def get_a_pair(self, index):
        with h5py.File(self.index_file, 'r') as db:
            i, j, label = json.loads(db['data'][index])
        return i, j, label
    
    def get_short_info_all_pairs(self):
        pairs = []
        with h5py.File(self.index_file, 'r') as db:
            for x in db['data']:
                i, j, score = json.loads(x)
                pairs.append((i, j, score))
        return pairs
    
    def get_full_info_all_pairs(self):
        pairs = self.get_short_info_all_pairs()
        full_pairs = []
        for i, j, score in pairs:
                query_ms = self.query_db[i]
                ref_ms = self.source_db[j]
                #full_pairs.append((i, j, query_ms[-2], ref_ms[-2], score))
                full_pairs.append((i, j, query_ms[1], ref_ms[1], query_ms[-2], ref_ms[-2], score))
        return full_pairs
    
    
    def get_full_info(self, index):
        i, j, label = self.get_a_pair(index)
        x = self.query_db[i]
        y = self.source_db[j]
        
        diff_pepmass = math.fabs(x[-1] - y[-1])
        mz_f, mz_mask = compute_feature_by_sim_matrix(np.array(x[2]), np.array(y[2]), diff_pepmass)
        loss_f, loss_mask = compute_feature_by_sim_matrix(np.array(x[3]), np.array(x[3]), diff_pepmass)
        return mz_f, mz_mask, loss_f, loss_mask, x, y, label
    
    def __getitem__(self, index):
        i, j, label = self.get_a_pair(index)
        x = self.query_db[i]
        y = self.source_db[j]
        
        diff_pepmass = math.fabs(x[-1] - y[-1])
        mz_f, mz_mask = compute_feature_by_sim_matrix(np.array(x[2]), np.array(y[2]), diff_pepmass)
        loss_f, loss_mask = compute_feature_by_sim_matrix(np.array(x[3]), np.array(x[3]), diff_pepmass)
        return mz_f, mz_mask, loss_f, loss_mask, label
    
        
class AllMSPairSet(MSPairSet):
    '''
    Set of all MS/MS spectrum pairs which are generated by pairing every query spectrum with all reference spectra.
    '''
    def __init__(self, source_db, query_db):
        self.source_db = source_db 
        self.query_db = query_db
        
        self.pairs = []
        for i in range(len(self.query_db)):
            for j in range( len(self.source_db)):
                self.pairs.append((i, j, 0))
        print(len(self.pairs))

    def __len__(self):
        return len(self.pairs)
    
    def get_a_pair(self, index):
        return self.pairs[index]
    
    def get_short_info_all_pairs(self):
        return self.pairs
    
    def get_targets(self, index):
        targets = [0 for _ in range(len(self.pairs))]
        return targets 
    
    
class RandomMSPairSet(MSPairSet):
    '''
    Set of MS/MS spectrum pairs which are generated by pairing every query spectrum with one reference spectra in each bin.
    '''
    def __init__(self, source_db, query_db, struct_sim_df):
        self.source_db = source_db 
        self.query_db = query_db
        
        self.struct_sim_df = struct_sim_df
        self.source_inchikey_index_dict = self.source_db.get_inchikey_indices()
        if source_db == query_db:
            self.query_inchikey_index_dict = self.source_inchikey_index_dict
        else:
            self.query_inchikey_index_dict = self.query_db.get_inchikey_indices()
        
        self.current_set = []
        self.randomize_ms_pairs()
        
    def _find_inchikey_in_range(self, query_key):
        source_keys = [] 
        start = 0 
        step = 0.1
        while (start < 1.0):
            #rows are source and columns are query
            end = start + step 
            matching_keys = self.struct_sim_df.index[(self.struct_sim_df[query_key] > start) 
                                                   & ((self.struct_sim_df[query_key] <= end))]
            if len(matching_keys) > 0: 
                source_keys.append(np.random.choice(matching_keys))
            start = end 
        return source_keys 
        
    def randomize_ms_pairs(self):
        
        self.current_set = []
        for query_key, index_group in self.query_inchikey_index_dict.items():
            source_keys = self._find_inchikey_in_range(query_key)
            
            for source_key in source_keys:
                i = np.random.choice(index_group)
                j = np.random.choice(self.source_inchikey_index_dict[source_key])
                score = self.struct_sim_df[query_key][source_key]
                self.current_set.append((i, j, score))
        return self.current_set
            
    def __len__(self):
        return len(self.current_set)
    
    def get_a_pair(self, index):
        return self.current_set[index]
        
    def get_short_info_all_pairs(self):
        return self.current_set
