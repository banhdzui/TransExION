'''
Created on 22 Feb 2021

@author: danhbuithi
'''
import sys 
import pandas as pd 

from common.command_args import CommandArgs
from common.io import save_data_in_hdf5_format

    
if __name__ == '__main__':
    config = CommandArgs({'in' : ('', 'Path of raw MS data file'),
                          'out' : ('', 'Path of output file')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    df = pd.read_csv(config.get_value('in'), header=0)
    spectrum_pairs = []
    print('loading spectra...')
    for i, row in df.iterrows():
        id1, id2, score = row[0], row[1], row[-1]
        spectrum_pairs.append((id1, id2, score)) 
    
    save_data_in_hdf5_format(config.get_value('out'), spectrum_pairs)
    