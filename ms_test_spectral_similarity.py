'''
Created on 17 Sep 2019

@author: danhbuithi
'''
import sys 
import random
import pandas as pd 

import torch
from torch.utils.data.dataloader import DataLoader

from common.command_args import CommandArgs

from lrp.functional import evaluate_spectral_similarity_measure
from lrp.data import C_MAX_PEAK_DIFF, C_NUM_DEFFECT_BIN
from lrp.data import  MSPairSet, MSDataset, mspair_collate_fn, AllMSPairSet
from lrp.model import relMSSimilarityModel


if __name__ == "__main__":
    
    config = CommandArgs({'db'  : ('', 'Path of db file'),
                          'query': ('', 'Path of query file'),
                          'pairs'   : ('', 'Path of testing spectrum pairs'),
                          'output'  : ('', 'Path of output file where save learned model'),
                          'model' : ('', 'Path of pre-trained model')
                          })        
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    manual_seed = random.randint(1, 10000)
    torch.manual_seed(manual_seed)
    nworkers = torch.get_num_threads()
    print(nworkers)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    net = relMSSimilarityModel(C_MAX_PEAK_DIFF+1, 
                               C_NUM_DEFFECT_BIN+2, 
                               hidden_dim=128, 
                               nclasses=1, 
                               dropout=0.1)
    net.load_state_dict(torch.load(config.get_value('model'), 
                                   map_location=device))
    net = net.double()
    
    source_db = MSDataset(config.get_value('db'))
    query_db = MSDataset(config.get_value('query'))
    test_index_file_name = config.get_value('pairs')
    
    if test_index_file_name == '':
        test_dataset = AllMSPairSet(source_db, query_db)
    else:
        test_dataset = MSPairSet(source_db, query_db, test_index_file_name)
    test_dataloader = DataLoader(test_dataset, 
                                 128, 
                                 shuffle=False, 
                                 num_workers=nworkers-1, 
                                 collate_fn=mspair_collate_fn)
   
    predict_values, true_values = evaluate_spectral_similarity_measure(test_dataloader, net, device)
    pair_info = test_dataset.get_full_info_all_pairs()
      
    save_result = []
    for x, y in zip(pair_info, predict_values):
        save_result.append((x[0], x[1], x[2], x[3], x[4], x[5], x[6], y))
    #save_result = sorted(save_result, key=lambda x: (x[0], y), reverse=True)
        
    df = pd.DataFrame(save_result, columns=['query_id', 'ref_id', 
                                            'query_key', 'ref_key', 
                                            'query_smiles', 'ref_smiles', 
                                            'true score', 'predict score'])
    df.to_csv(config.get_value('output'), index=False) 
    

