# Overview 
This release provides a fully functional 1D subsidence modeling framework for 39 benchmark sites within the Central Valley of California, built using MODFLOW 6 with the CSUB package. The repository integrates PEST-PyEMU for ensemble-based data assimilation and model calibration, enabling robust estimation of critical heads and projection of future subsidence under various groundwater management scenarios.

# Key Features:

- MODFLOW 6-CSUB Implementation: Simulates both elastic and inelastic compaction of aquifer-system interbeds, with support for delay and no-delay beds. In this release, the multiple clay interbeds identified in the site’s lithology are explicitly represented as individual interbeds, rather than being combined into a single equivalent layer as in the main branch.

- Critical Head Estimation: Automated analysis to determine the critical groundwater levels below which permanent (inelastic) subsidence occurs.

- Calibration Workflow: A complete PEST-PyEMU calibration pipeline (via workflow.py and ies-functions.py) to minimize error between simulated and observed subsidence using:

 - InSAR data

 - Extensometer data

 - Historical groundwater levels
# Pre-processing and Data Handling: 
- Scripts (prep_data.py, model_functions.py) for:

	- Cleaning and resampling observed groundwater data

	- Generating GHB head boundaries with autocorrelations and vertical bias adjustments

	- Grouping interbeds by thickness quantiles

# Repository Contents

- workflow.py: Orchestrates the modeling workflow (pre-processing, calibration, prediction).

- ies-functions.py: Implements data assimilation and objective function calculations.

- model_functions.py: Contains MODFLOW 6-CSUB and GHB setup utilities.

- prep_data.py: Prepares and processes observational data for calibration and validation.

# Applications

- SGMA Compliance: Supports Sustainable Management Criteria (SMC) development.

- Risk Assessment: Provides insight into subsidence impacts on infrastructure under different management strategies.

- Research and Teaching: Demonstrates the capabilities of MODFLOW 6-CSUB for both site-specific and regional studies.


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
