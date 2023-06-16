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
**Input Data**: To work with large scale dataset, the input data files are converted into h5py format. Our library supports a function to convert a MGF format file into h5py format. To do the conversion, run the script ```ms_convert_mgf2h5py.py``` with parameters. For example,

<pre><code class="language-python"> python ms_convert_mgf2h5py.py --mgf [a MGF file] --out [output file] </code></pre>
  
**Training model**
  
**Testing model**
  
**Explanation**

