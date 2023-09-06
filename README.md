# PyTorch implementation of TransExION: A Transformer - based Explainable similarity metric for IONS in Tandem Mass Spectrometry
## Introduction
TransExION is a supervised learning method estimating ***spectral similarity*** between MS/MS spectra that are strongly correlated to their structural similarity. It can be used in spectral library search to find structural analogues. TransExION is based on Transformer architecture and provides a ***post hoc explanation*** for its outcome in order to reveal the relationship between fragments.

**The workflow of TransExION**
<div>
<image src="https://github.com/banhdzui/TransExtION/assets/37836357/0941f8c1-dc53-4eef-addd-17152ec79c0e">
</div>
  
**TransExION can be used in spectral library search to find structural analogues**
<div>
<image src="https://github.com/banhdzui/TransExtION/assets/37836357/17e31fc7-03aa-4438-ba66-136ae6f3f5d1" width="450">
</div>
  
**TransExION provides a post-hoc explanation**

It is a heatmap indicating the association between the query fragments and reference fragments. Below is an example of the heatmap and their structure. Each heatmap cell shows the query product ion (the first value in the brackets), reference product ion(s) and their mass difference (the second value in the brackets).   
  <div>
<image src="https://github.com/banhdzui/TransExtION/assets/37836357/358ca639-b82f-4537-ac34-76f7ac1ff66f" width="200">
<image src="https://github.com/banhdzui/TransExtION/assets/37836357/944c741a-8eb8-4bfd-af8c-6eb3059f8859" width="500">
  </div>
  
## Requirements  
* Python 3.x 
* PyTorch 
* RDKit 
* Matplotlib 
* h5py
* pyteomics

## Usage
**Input Data**: 

To work with large scale dataset, the input data files are converted into h5py format. Our library supports a function to convert a MGF format file into h5py format. To do the conversion, run the script ```ms_convert_mgf2h5py.py``` with parameters.

```
python ms_convert_mgf2h5py.py --mgf [input file] --out [output file]
```
Required arguments:
```
--mgf  Path to the input file which is in mgf format.

--out   Path to the output file which is in h5py format.
```
  
**Training model**

A TransExION model can be trained by running the script ```ms_train_spectal_similarity.py``` with parameters. 

```
python ms_train_spectal_similarity.py --db [training spectrum file] --query [validation spectrum file] --db-ref [training structural similarity file] --query-ref [validation structural similarity file] --batchsize [batch size] --lr [learning rate] --decay-weight [decay weight] --output [model file] --log [log file]
```

Required arguments: 
|Parameter  |Default  |Description  |
|:-----------|:--------|:------------|
|--db  |  |Path to a MS/MS spectrum file which is used for training (h5py format)| 
|--query  |  |Path to a MS/MS spectrum file which is used for validation (h5py format)|
|--db-ref  |  |Path to the file containing structural similarity between the training MS/MS spectra|
|--query-ref  |  |Path to the file containing structural similary between the validation MS/MS spectra| 
|--batchsize  |64  |Batch size for training|
|--lr  |1e-4  |Learning rate for training|
|--decay-weight  |0.0  |Decay weight for training|
|--output  |  |Path to output file saving the trained model|
|--log  |  | Path to a log file|

To compute the structural similarity matrix between MS/MS spectra, the script ```ms_compute_structural_similarity.py``` can be used. 

The data we used to train the model can be found in [zenodo link](https://zenodo.org/record/8175528). Given a training file, ```GNPS_MassBank_train.mgf```, and a validation file, ```GNPS_MassBank_val.mgf```, we can train a TransExION model as follows: 

Convert mgf files into h5py format files:
```
python ms_convert_mgf2h5py.py --mgf GNPS_MassBank_train.mgf --out GNPS_MassBank_train.db

python ms_convert_mgf2h5py.py --mgf GNPS_MassBank_val.mgf --out GNPS_MassBank_val.db
```
Compute the structral similarity matrices: 
```
python ms_compute_structural_similarity.py --db GNPS_MassBank_train.mgf --query GNPS_MassBank_train.mgf --out GNPS_MassBank_train.ref

python ms_compute_structural_similarity.py --db GNPS_MassBank_train.mgf --query GNPS_MassBank_val.mgf --out GNPS_MassBank_val.ref
```
Train the TransExION model:
```
python ms_train_spectal_similarity.py --db GNPS_MassBank_train.db --query GNPS_MassBank_val.db --db-ref GNPS_MassBank_train.ref --query-ref GNPS_MassBank_val.ref --batchsize 64 --lr 0.0001 --decay-weight 0.0 --output GNPS_MassBank.ms.model --log training_log.txt
```

**Testing model**
  
To compute spectral similarity between spectra using a trained TransExION model, the script ```ms_test_spectral_similarity.py``` can be used with parameters

```
python ms_test_spectral_similarity.py --db [reference spectrum file] --query [query spectrum file] --pairs [spectrum pair id file] --model [model file] --output [result file]
```
Required arguments:
|Parameter  |Default  |Description  |
|:-----------|:--------|:------------|
|--db  |  |Path to a MS/MS spectrum file which is used as spectral library (h5py format)| 
|--query  |  |Path to a MS/MS spectrum file which is used as query (h5py format)|
|--pairs  |empty  |Path to the file containing the indices of spectrum pairs to estimate spectral similarity|
|--model  |  |Path to the pre-trained TransExION model|
|--output  |  | Path to output file saving the result|

If the parameter ```--pairs``` is empty, then every query spectrum is paired with every reference spectra to compute spectral similarity. The ```spectrum pair id file``` should also be in h5py format. You can convert a csv file into h5py format using the script ```ms_covert_csv2h5py.py```.

For example:
Convert the query file in mgf format into h5py format file:
```
python ms_convert_mgf2h5py.py --mgf GNPS_MassBank_test.mgf --out GNPS_MassBank_test.db
```
Compute spectral similarity between the query spectra and the reference spectra
```
python ms_test_spectral_similarity.py --db GNPS_MassBank_train.db --query GNPS_MassBank_test.db  --model GNPS_MassBank.ms.model --output GNPS_MassBank_result.csv
```
**Explanation**

The post-hoc explanation is demonstrated in the script ```ms_explain_spectral_similarity.py```. The script can be run with parameters.
```  
python ms_explain_spectral_similarity.py --db [reference spectrum file] --query [query spectrum file] --pairs [spectrum pair id file] --model [model file] --output [output folder]
```
The required arguments are same as the testing, but the parameter ```--output``` is the path to a directory where contains generated heatmap files for every pair.
