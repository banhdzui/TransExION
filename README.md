# PyTorch implementation of TransExtION: Transformer based Explainable similarity metric for IONS 
## Introduction
TransExtION is a supervised learning method estimating ***spectral similarity*** between MS/MS spectra that are strongly correlated to their structural similarity. It can be used in spectral library search to find structural analogues. TransExtION is based on Transformer architecture and provides a ***post hoc explanation*** for its outcome in order to reveal the relationship between fragments.

The workflow of TransExtION
<div>
<image src="https://github.com/banhdzui/TransExtION/assets/37836357/0941f8c1-dc53-4eef-addd-17152ec79c0e">
</div>
  
TransExtION can be used in spectral library search to find structural analogues
<div>
<image src="https://github.com/banhdzui/TransExtION/assets/37836357/17e31fc7-03aa-4438-ba66-136ae6f3f5d1" width="450">
</div>
  
TransExtION provides a post-hoc explanation. It shows the heatmap indicating the association between the query fragments and reference fragments. Below is an example of the heatmap and their structure. Each heatmap cell shows the query product ion (the first value in the brackets), reference product ion(s) and their mass difference (the second value in the brackets).   
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
 
<pre><code class="language-python"> python ms_convert_mgf2h5py.py --mgf [a MGF file] --out [output file] </code></pre>
  
**Training model**

A TransExtION model can be trained by running the script ```ms_train_spectal_similarity.py``` with parameters. 
  
<pre><code class="language-python"> python ms_train_spectal_similarity.py --db [training spectrum file] --query [validation spectrum file] --db-ref [training structural similarity file] --query-ref [validation structural similarity file] --batchsize [batch size] --lr [learning rate] --decay-weight [decay weight] --output [model file] --log [log file] </code></pre>
  
The ```trainiing spectrum file``` and ```validation spectrum file``` should be in h5py format. To convert a MGF format file to a h5py format file, see the Input Data section.

```Structural similarity files``` contain the structural similarity matrix among spectra which can be computed by using the script ```ms_compute_structural_similarity.py```. 
  
**Testing model**
  
To compute spectral similarity between spectra using a trained TransExtION model, the script ```ms_test_spectral_similarity.py``` can be used. It pairs the spectra in the query file with the ones in the reference file and then computes their spectral similarity.
  
<pre><code class="language-python"> python ms_test_spectral_similarity.py --db [reference spectrum file] --query [query spectrum file] --pairs [spectrum pair id file] --model [model file] --output [result file]</code></pre>

The ```reference spectrum file``` and ```query spectrum file``` should be in h5py format. To convert a MGF format file to a h5py format file, see the Input Data section.

The pairing depends on the param ```pairs```. This param enumerated the spectrum pairs that the users would like to estimate the spectral similarity. If it is empty then every query spectrum is paired with every reference spectra. The ```spectrum pair id file``` should also be in h5py format. You can convert a csv file into h5py format using the script ```ms_covert_csv2h5py.py```.
  
**Explanation**

The post-hoc explanation is demonstrated in the script ```ms_explain_spectral_similarity.py```. The script can be run with parameters.
  
<pre><code class="language-python"> python ms_explain_spectral_similarity.py --db [reference spectrum file] --query [query spectrum file] --pairs [spectrum pair id file] --model [model file] --output [output folder]</code></pre>
  
The script pairs the spectra in the query file with the ones in the reference file and then do post-hoc explanation for the pairs that the model estimates as highly similar in structure. The output are heatmap files which are generated in output folder.
