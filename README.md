# Getting Started
1. Install Miniforge

   If you do not have a [Miniforge](https://github.com/conda-forge/miniforge), [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#), [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install), or [anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) python distribution on your computer install miniforge for your operating system from:

   https://github.com/conda-forge/miniforge 

2.  Open a terminal/command prompt in the root directory of the repo.

3. Install the project environment in your python environment:

   a. If you have a minforge or a miniconda python installation:
      ```
      mamba env create -f environment.yml
      ```

   b. If you have a miniconda or a miniconda python installation without mamba:
      ```
      conda env create -f environment.yml
      ``` 

4.  Activate the project environment:
    ```
    conda activate dwr_subsidence
    ```
5.  Run the project script:
    ```
    python DWR.py
    ```