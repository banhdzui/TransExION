'''
Created on 17 Sep 2019

@author: danhbuithi
'''
import sys 
import random
import pandas as pd 
from collections import Counter

import torch
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader

from common.command_args import CommandArgs
from common.pytorchtools import EarlyStopping
from common.io import create_log_file

from lrp.data import C_MAX_PEAK_DIFF, C_NUM_DEFFECT_BIN
from lrp.data import MSDataset, mspair_collate_fn, RandomMSPairSet
from lrp.model import relMSSimilarityModel
from lrp.functional import train_spectral_similarity_measure
from lrp.functional import test_spectral_similarity_measure

def get_sample_weights(targets):
    
    samples_per_class = dict(Counter(targets))
    print(samples_per_class)
    n = len(targets)
    class_weights = {c: (1-v/n) for c, v in samples_per_class.items()}
    print(class_weights)
    
    sample_weights = [class_weights[x] for x in targets]
    return sample_weights

if __name__ == "__main__":
    
    config = CommandArgs({'db_ref'   : ('', 'Path of the file containing structural similarity among training spetra'),
                          'query_ref'     : ('', 'Path of the file containing structral similarity among validation spectra'),
                          'db'      : ('', 'Path of database file'),
                          'query'   : ('', 'Path of query (val) data file'),
                          'batchsize'   : (64, 'Input batch size'),
                          'lr'      :   (1e-4, 'Learning rate'),
                          'decay_weight'    :(0, 'Weight of decay'),
                          'nloop'   : (100, 'Number of training iterations'),
                          'output'  : ('', 'Path of output file where save learned model'),
                          'model' : ('', 'Path of pre-trained model'),
                          'log' : ('', 'Path of log file'),
                          'name'  : ('', 'Log name')
                          })        
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    manual_seed = random.randint(1, 10000)
    torch.manual_seed(manual_seed)
    log_name = config.get_value('name')
    logger_file = create_log_file(log_name, config.get_value('log'))
    
    '''
    Load params values
    '''
    nworkers = torch.get_num_threads()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    logger_file.info('used device ' + str(device))
    
    logger_file.info('load arguments...')
    batch_size = int(config.get_value('batchsize'))
    weight_decay = float(config.get_value('decay_weight'))
    niter = int(config.get_value('nloop'))
    
    lr = float(config.get_value('lr'))
    logger_file.info('lr ' + str(lr))
    logger_file.info('weight decay ' + str(weight_decay))
    
    logger_file.info('Creating model....')
    net = relMSSimilarityModel(C_MAX_PEAK_DIFF+1, C_NUM_DEFFECT_BIN+2, hidden_dim=128, nclasses=1, dropout=0.1)

    net.to(device)
    net = net.double()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    logger_file.info('Loading training data ....')
    source_db = MSDataset(config.get_value('db'))
    train_ref_df = pd.read_pickle(config.get_value('db_ref'))
    train_dataset = RandomMSPairSet(source_db, source_db, train_ref_df)
    training_stopper = EarlyStopping(patience=10, verbose=True, delta=0, path=config.get_value('output'))
    
    val_db = MSDataset(config.get_value('query'))
    val_ref_df = pd.read_pickle(config.get_value('query_ref'))
    val_dataset = RandomMSPairSet(source_db, val_db, val_ref_df)
    val_dataloader = DataLoader(val_dataset, batch_size, collate_fn=mspair_collate_fn)
    
    logger_file.info('Start training similarity...')
    criterion = nn.MSELoss()
    if config.get_value('model'):
        net.load_state_dict(torch.load(config.get_value('model'), map_location=device))
        val_loss = test_spectral_similarity_measure(val_dataloader, net, criterion, device)
        training_stopper(val_loss, net, logger_file)
        
    for epoch in range(niter):
        print(len(train_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size, 
                                  shuffle=True, num_workers=nworkers-1,
                               collate_fn=mspair_collate_fn)
        train_spectral_similarity_measure(epoch, niter, train_dataloader, net, optimizer, criterion, device, logger_file)
        
        val_dataloader = DataLoader(val_dataset, batch_size, collate_fn=mspair_collate_fn)
        val_loss = test_spectral_similarity_measure(val_dataloader, net, criterion, device)
        
        training_stopper(val_loss, net, logger_file)
        if training_stopper.early_stop: break 
        
        train_dataset.randomize_ms_pairs()
        