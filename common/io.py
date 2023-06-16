'''
Created on 24 Feb 2021

@author: danhbuithi
'''
import pickle 
import json 
import logging
import h5py


def load_pickle(file_name):
    '''
    Open and load content of a pickle file
    '''
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def dump_dictionary(dictionary, filename):
    '''
    Save an object into a pickle file
    '''
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
        
def save_in_json_format(file_name, data):
    '''
    Save an object into a json-format file
    '''
    with open(file_name, 'w') as file_writer:
        json.dump(data, file_writer)
        
def load_json_file(file_name):
    '''
    Open and load an object from a json-format file
    '''
    with open(file_name, 'r') as file_reader:
        return json.load(file_reader)
        
def save_data_in_hdf5_format(file_name, data_set):
    '''
    Save a dataset into hdf5 format
    '''
    file_writer = h5py.File(file_name, 'w')
    dt = h5py.string_dtype(encoding='ascii')
    dset = file_writer.create_dataset('data', (len(data_set),), dtype=dt, compression='gzip')
    for i, x in enumerate(data_set):
        dset[i] = json.dumps(x)
    file_writer.close()
    
def create_log_file(log_name, file_name):
    '''
    Create a log file (INFO level)
    '''
    logger_file = logging.getLogger(log_name)
    logger_file.setLevel(logging.INFO)
    fh = logging.FileHandler(file_name)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger_file.addHandler(fh)
    return logger_file
