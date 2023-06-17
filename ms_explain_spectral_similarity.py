'''
Created on 17 Sep 2019

@author: danhbuithi
'''
import sys 
import random
import torch

from common.command_args import CommandArgs

from lrp.data import C_MAX_PEAK_DIFF, C_NUM_DEFFECT_BIN  
from lrp.data import MSPairSet, MSDataset, AllMSPairSet
from lrp.model import relMSSimilarityModel
from lrp.functional import explain_spectral_similarity
from lrp.explanator import register_forward_hook_4_model


if __name__ == "__main__":
    
    config = CommandArgs({'db'  : ('', 'Path of db file'),
                          'query': ('', 'Path of query file'),
                          'pairs'   : ('', 'Path of testing spectrum pairs'),
                          'model' : ('', 'Path of pre-trained model'),
                          'output' : ('', 'Path of output folder')
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
    net.load_state_dict(torch.load(config.get_value('model'), map_location=device))
    net = net.double()
    
    source_db = MSDataset(config.get_value('db'))
    query_db = MSDataset(config.get_value('query'))
    test_index_file_name = config.get_value('pairs')
    if test_index_file_name == '':
        test_dataset = AllMSPairSet(source_db, query_db)
    else:
        test_dataset = MSPairSet(source_db, query_db, test_index_file_name)
    
    register_forward_hook_4_model(net)
    output_folder = config.get_value('output')
    explain_spectral_similarity(test_dataset, net, device, output_folder)
    
