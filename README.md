# Analysis-of-methylated-and-unmethylated-reads
My work on distinguishing methylated and unmethylated reads summer 2018.
The full documentation and research summary is in the .ipynb file Progress report. 

The requirements to run everything are:

Python packages:
  Matplotlib 
  Numpy 
  scikit-learn
  Scipy
  fastdtw
  Cython
  Numba
  Pytorch
  h5py
  tslearn
  
Jupyterlab and and this extension:
https://ipywidgets.readthedocs.io/en/stable/user_install.html

This also requires node.js.

In addition, for the various FDBA implementations to run fast, a c compiler is required, and for pytorch training to run fast, cuda 9.2 support is required.

To download all of this, I recommend using an linux os with gcc as the c compiler, installing anaconda, then installing through it Pytorch, fastdtw, h5py, tslearn and node.js, install the jupyter lab widget extension through the above link.
