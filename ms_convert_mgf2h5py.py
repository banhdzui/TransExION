'''
Created on 22 Feb 2021

@author: danhbuithi
'''
import sys 
import glob 

from common.command_args import CommandArgs
from common.io import save_data_in_hdf5_format
from spectrum.io import load_mgf_file, convert_raw2refined_spectra, load_mgf_files

    
if __name__ == '__main__':
    config = CommandArgs({'mgf' : ('', 'Path of raw MS data file'),
                          'folder': ('', 'Path of the folder containing ms data files'),
                          'out' : ('', 'Path of output file')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
            
    if config.get_value('mgf') != '':
        ms_data = load_mgf_file(config.get_value('mgf'), mol_id_key=None, use_drug = False)
    else:
        folder_path = config.get_value('folder')
        folder_path = folder_path + '/*.mgf'
        ms_data = load_mgf_files(glob.glob(folder_path), use_drug=False, mol_id_key=None)
        
    transformed_data = convert_raw2refined_spectra(ms_data)
    print(len(transformed_data))
    save_data_in_hdf5_format(config.get_value('out'), transformed_data)
    