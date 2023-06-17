'''
Created on 22 Feb 2021

@author: danhbuithi
'''
import sys 
import torch 
import numpy as np
import pandas as pd 
from rdkit import Chem, DataStructs

from common.command_args import CommandArgs
from spectrum.io import load_mgf_file
from multiprocessing import Pool

def _group_spectra_by_inchikey(db):
    inchikey_smiles = {}
    inchikey_indices = {}
    for s in db: 
        key = s.get_short_inchikey()
        mid = s.mid
        
        if key not in inchikey_smiles: 
            inchikey_smiles[key] = s.smiles
            inchikey_indices[key] = []
            
        inchikey_indices[key].append(mid)
    return list(inchikey_smiles.items()), inchikey_indices

def compute_tanimoto_score(q_smiles, s_smiles):
    q_mol = Chem.MolFromSmiles(q_smiles)
    s_mol = Chem.MolFromSmiles(s_smiles)
    
    if q_mol is None or s_mol is None: return -1
    
    q_fps = Chem.RDKFingerprint(q_mol)
    s_fps = Chem.RDKFingerprint(s_mol)
    v = DataStructs.FingerprintSimilarity(q_fps, s_fps, metric = DataStructs.TanimotoSimilarity)
    return v 

def _compute_structure_similarity(args):
    q_key, q_smiles, smiles_db = args
    
    similarity_values = []
    for _, s_smiles in smiles_db: 
        v = compute_tanimoto_score(q_smiles, s_smiles)
        similarity_values.append(v)
    return q_key, similarity_values      
    

if __name__ == '__main__':
    config = CommandArgs({'db' : ('', 'Path of MGF data file'),
                          'query'  : ('', 'Path of input file'),
                          'out' : ('', 'Path of output file')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    db = load_mgf_file(config.get_value('db'), mol_id_key = None)
    query = load_mgf_file(config.get_value('query'), mol_id_key = None)
    
    nworkers = torch.get_num_threads()
    smiles_query, _ = _group_spectra_by_inchikey(query) # 
    smiles_db, _ = _group_spectra_by_inchikey(db)
    
    print('computing similarity matrix...')
    params = []
    for q_key, q_smiles in smiles_query:
        params.append((q_key, q_smiles, smiles_db))
       
    with Pool(nworkers) as pool:
        values = pool.map(_compute_structure_similarity, params)
        struct_similarity_matrix = {q_key : x for q_key, x in values }
        
    index = [x[0] for x in smiles_db]
    columns = []
    values = []
    for query_key, scores in struct_similarity_matrix.items(): 
        columns.append(query_key)
        values.append(scores)
        
    values = np.transpose(np.array(values))
    df = pd.DataFrame(values, columns = columns, index=index)
    df.to_pickle(config.get_value('out'))