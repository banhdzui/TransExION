'''
Created on 29 Apr 2021

@author: danhbuithi
'''

import math 
import numpy as np 

from pyteomics import mgf 


def _remove_dense_peaks(mz, intensity, radius=1.5, d = 3):
    '''
    Remove peaks which are very close to each other, keep the highest peaks.
    '''
    sorted_indices = np.argsort(intensity)[::-1]
    checked_intensity = [v for v in intensity]
    n = len(intensity)
    
    for i in sorted_indices:
        if checked_intensity[i] == 0: continue 
        j = i - 1
        ij = i
        while (j >= 0 and 
               (mz[ij] - mz[j]) < radius and 
               intensity[i] >= intensity[j] and 
               (mz[i] - mz[j]) < d):
            checked_intensity[j] = 0
            ij = j
            j -= 1 
        j = i + 1
        ij = i
        while (j < n and 
               (mz[j] - mz[ij]) < radius and 
               intensity[i] >= intensity[j] and 
               (mz[j] - mz[i]) < d): 
            checked_intensity[j] = 0
            ij = j
            j += 1 
    
    reduced_mz = []
    reduced_intensity = []
    
    for i, v in enumerate(checked_intensity):        
        if v == 0:
            continue 
        
        reduced_mz.append(mz[i])
        reduced_intensity.append(v)
    return reduced_mz, reduced_intensity
            
        
class MSObject(object):
    '''
    MSObject is a MS/MS spectrum with some basic information.
    It stores the following information: id, inchikey, smiles, precursor mass, formula, mz, and intensity
    '''
    def __init__(self, mid, inchikey, smiles, pepmass, formula, mz, intensity):
        self.mid = mid
        self.inchikey = inchikey
        
        self.smiles = smiles 
        self.pepmass = pepmass
        self.formula = formula
         
        self.mz = mz 
        self.intensity = intensity 
        
        
    def get_short_inchikey(self):
        '''
        Returns the first 14 characters in INCHIKEY (conectivity information)
        '''
        return self.inchikey[:14]

    def normalize(self):
        '''
        Normalize the intensity of the spectrum
        '''
        r = np.array(self.intensity)
        self.intensity = r/np.max(r)
        
    def remove_noise_peaks(self, min_peak=10, max_peak = 1000, threshold=0.001, use_sqrt=False):
        '''
        Remove peaks which are out of range [min_peak, max_peak] and are lower than the threshold in intensity
        '''
        reduced_mz = []
        reduced_intensity = []
        
        for i, x in enumerate(self.intensity):
            if x < threshold: continue 
            if self.mz[i] < min_peak or self.mz[i] > max_peak: continue 
             
            reduced_mz.append(self.mz[i])
            if use_sqrt == True:
                reduced_intensity.append(math.sqrt(x))
            else: 
                reduced_intensity.append(x)
                
        self.mz = reduced_mz 
        self.intensity = reduced_intensity      
        
    def remove_dense_peaks(self, radius=1.5, d = 3):
        '''
        Remove peaks which are very close to each other, only keep the highest peak.
        '''
        reduced_mz, reduced_intensity = _remove_dense_peaks(self.mz, self.intensity, radius, d)
        self.mz = reduced_mz 
        self.intensity = reduced_intensity
        
    def get_neutral_loss(self):
        x = np.array(self.mz)
        x = (x[-1] - x)[::-1]
        return list(x[1:])
    
def load_mgf_file(file_name, use_drug = False, encoding='utf-8', mol_id_key=None, prefix=''):
    '''
    Loads MS/MS spectra from a mgf file
    '''
    ms_data = []
    with mgf.MGF(file_name, encoding=encoding) as reader: 
        for spectrum in reader: 
            header = spectrum['params']
            if (use_drug == False) and ('db' in header) and (header['db'] == 'JANSSEN'): 
                continue
            
            smiles = header['smiles']
            #mol = Chem.MolFromSmiles(smiles)
            #if mol is None:
            #    print(smiles) 
            #    continue 
            formula = ''
            if 'formula' in header: 
                formula = header['formula']
                
            inchikey = ''
            if 'inchikey' in header:
                inchikey = header['inchikey']
            
            pepmass = header['pepmass'][0]
            mz = list(spectrum['m/z array'])
            intensity = list(spectrum['intensity array'])
            
            if mol_id_key is not None: 
                mol_id = prefix + header[mol_id_key]
            else: 
                mol_id = len(ms_data)
    
            o = MSObject(mol_id, inchikey, smiles, pepmass, formula, mz, intensity)
            ms_data.append(o)
            
    return ms_data 

def load_mgf_files(file_names, use_drug = False, encoding='utf-8', mol_id_key=None, prefix=''):
    '''
    Loads MS/MS spectra from mgf files
    '''
    ms_data = []
    for file_name in file_names:
        with mgf.MGF(file_name, encoding=encoding) as reader: 
            for spectrum in reader: 
                header = spectrum['params']
                if (use_drug == False) and ('db' in header) and (header['db'] == 'JANSSEN'): 
                    continue
                
                smiles = header['smiles']
                
                formula = ''
                if 'formula' in header: 
                    formula = header['formula']
                
                inchikey = ''
                if 'inchikey' in header:
                    inchikey = header['inchikey']
                
                pepmass = header['pepmass'][0]
                
                mz = list(spectrum['m/z array'])
                intensity = list(spectrum['intensity array'])
                
                if mol_id_key is not None: 
                    mol_id = prefix + header[mol_id_key]
                else: 
                    mol_id = len(ms_data)
        
                o = MSObject(mol_id, inchikey, smiles, pepmass, formula, mz, intensity)
                ms_data.append(o)
        print(len(ms_data))   
    return ms_data

def convert_raw2refined_spectra(ms_data):
    npeaks = 0 
    transformed_data = []
    for x in ms_data:
        x.normalize()
        x.remove_noise_peaks()
        x.remove_dense_peaks()
        if len(x.mz) > npeaks: npeaks = len(x.mz)
        
        neutral_loss = x.get_neutral_loss()
        transformed_data.append((x.mid, x.get_short_inchikey(), x.mz, neutral_loss, x.intensity, x.smiles, x.pepmass))
    
    return transformed_data
    