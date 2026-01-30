## **Source_data**

The **Source_data** directory contains all primary model input files required to run the 1D MODFLOW 6–CSUB simulations. These files define the hydrostratigraphy, groundwater-level forcing, subsidence observations, and parameter values used during model calibration and scenario evaluation.

- **CH_forecast spreadsheet**
  - Contains groundwater-level input files for predictive simulations evaluating subsidence relative to the estimated critical head.
  - Used to assess future subsidence under groundwater levels maintained at or above the modeled critical head threshold.

- **Lithology spreadsheet**
  - Includes site-specific hydrostratigraphic information for each model location.
  - Defines aquifer and aquitard units, unit names, top and bottom elevations, and generalized soil texture classifications (sand or clay).
  - Used to define vertical layering, interbed placement, and delay-interbed systems in the CSUB package.

- **obs_data spreadsheet**
  - Contains measured groundwater-level observations for each modeled aquifer unit.
  - Time series are derived from nearby monitoring wells and used to construct long-term groundwater-level inputs.
  - Multiple wells may be combined to form composite hydrographs where needed.

- **par_data spreadsheet**
  - Contains initial (prior) parameter values used at the start of data assimilation.
  - Defines pre-calibration estimates for key groundwater-flow and compaction parameters.
  - Represents assumed uncertainty in model inputs prior to conditioning on subsidence observations.

- **scenario_data spreadsheet**
  - Includes groundwater-level inputs for the historical low groundwater-level scenario.
  - Represents conditions where groundwater levels decline to historic lows and remain constant.
  - Used to evaluate potential future subsidence under continued low-head management.

- **scenario_data_2015 spreadsheet**
  - Contains groundwater-level inputs for the 2015 water-level scenario.
  - Represents recovery to 2015 conditions.
  - Used to estimate future subsidence associated with legacy pumping impacts.

- **sub_data spreadsheet**
  - Contains observed subsidence measurements used for model calibration.
  - Includes time series from leveling surveys, InSAR, GPS, and extensometers.
  - Data are processed and registered to a common vertical datum where applicable.
  - Serves as the primary calibration target in the data assimilation workflow.

---

## **Interbed Results Spreadsheet (ib_results)**

The **ib_results** spreadsheet summarizes final calibrated parameters for all modeled clay interbeds across aquifer units and model layers.

- Includes calibrated values of:
  - Vertical hydraulic conductivity
  - Elastic and inelastic skeletal storage parameters
  - Representative interbed thicknesses
- Provides a concise summary of depth-dependent compaction properties.
- Can be used to inform parameterization of regional groundwater-flow models.

---

## **Preprocessing Script (prep_data.py)**

The **prep_data.py** script automates preprocessing of model input data.

- Reads:
  - Groundwater-level observations
  - Lithologic information
  - Subsidence measurements
  - Parameter files
- Formats inputs into standardized structures required by:
  - MODFLOW 6–CSUB
  - The data assimilation framework

---

## **Outputs**

This folder contains spreadsheets summarizing modeled and observed subsidence results.

- Includes simulated cumulative subsidence for:
  - Calibrated (base) model
  - Critical Head
  - Critical Head +20 ft
  - Critical Head +50 ft
  - Historical Low
  - 2015 water-level conditions
- Includes observed subsidence time series for comparison with modeled results.
