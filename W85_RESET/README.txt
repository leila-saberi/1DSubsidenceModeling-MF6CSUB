Folder Structure and Contents/

This folder contains the full set of input data, preprocessing script, and model outputs used to develop and calibrate the one-dimensional MODFLOW 6–CSUB subsidence models. The directory structure is organized to separate raw and scenario-specific input data, simulation outputs, and calibrated parameter outputs in a transparent and reproducible manner.

Source_data:/
The Source_data directory contains all primary model input files required to run the 1D MODFLOW 6–CSUB simulations. These files define the hydrostratigraphy, groundwater-level forcing, subsidence observations, and parameter values used during model calibration and scenario evaluation.

- The CH_forecast spreadsheet contains groundwater-level input files used for predictive simulations designed to evaluate subsidence response relative to the estimated critical head. These inputs are used to assess future subsidence under groundwater levels maintained at and above the modeled critical head threshold.

- The lithology spreadsheet includes site-specific hydrostratigraphic information for each model location. This file define aquifer and aquitard units, including unit names, top and bottom elevations, and generalized soil texture classifications (sand or clay). These data are used to define vertical layering, interbed placement, and delay-interbed systems in the CSUB package.

- The obs_data spreadsheet contains measured groundwater-level observations associated with each modeled aquifer unit. These time series are derived from nearby monitoring wells and are used to construct long-term groundwater-level inputs for the 1D models. Where necessary, observations from multiple wells are combined to form composite hydrographs representative of local stress conditions.

- The par_data spreadsheet contains the initial (prior) parameter values used at the start of the data assimilation process. These files define pre-calibration estimates for key groundwater-flow and compaction parameters, including vertical hydraulic conductivity and skeletal storage properties. These priors represent the assumed uncertainty in model inputs before conditioning on subsidence observations.

- The scenario_data spreadsheet includes groundwater-level input files used for the historical low groundwater-level scenario. These inputs represent conditions in which groundwater levels decline to historically low values and are then held constant to evaluate potential future subsidence under continued low-head management.

- The scenario_data_2015 spreadsheet contains groundwater-level input files for the 2015 water-level scenario. These files represent recovery of groundwater levels to 2015 conditions and are used to estimate future subsidence associated with legacy pumping impacts prior to 2015.

- The sub_data spreadsheet contains observed subsidence measurements used for model calibration. These data include time series derived from leveling surveys, InSAR, GPS, and extensometer records, processed and registered to a common vertical datum where applicable. These observations serve as the primary calibration targets in the data assimilation workflow.

Interbed Results Spreadsheet (ib_results):
The ib_results spreadsheet summarizes the final calibrated parameters for all modeled clay interbeds across each aquifer unit and model layer. This file includes calibrated values of vertical hydraulic conductivity, elastic and inelastic skeletal storage parameters, and representative interbed thicknesses. These results provide a concise summary of how compaction properties vary with depth and can be used to inform parameterization of regional groundwater-flow models.

Preprocessing Script (prep_data.py):
The prep_data.py script is used to automatically preprocess model input data as part of the modeling workflow. This script reads raw groundwater-level observations, lithologic information, subsidence measurements, and parameter files, and formats them into the standardized input structures required by MODFLOW 6–CSUB and the data assimilation framework.

Outputs:
This folder contains a spreadsheet summarizing simulated cumulative subsidence for the calibrated (base) model and future groundwater-level scenarios, including Critical Head, Critical Head +20 ft, Critical Head +50 ft, Historical Low, and 2015 water-level conditions. Observed subsidence time series are included for comparison with modeled results.
