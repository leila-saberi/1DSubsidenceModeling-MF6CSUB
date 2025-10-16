
import workflow
import pandas as pd
from importlib import reload
from datetime import date
import random

today = date.today()
date_string = today.strftime("%m%d%Y")

def main(
        sites=None,
        num_reals= 500,
        monthly_num_workers= 120,
        monthly_noptmax=35,
        use_focus_weights=True,
        use_obs_diff=True,
        use_delay=True,
        plot=True,
        port=None,
        scenario_subset = 100
):
    """
    This function runs the workflow with history matching and scenarios.

    Parameters
    ----------
    sites : str, optional
        Site name, by default None. If None, sites T88, J88, H201, GWM_14, and 341.804 are evaluated.
    num_reals : int, optional
        Number of realizations for running the ensemble, by default 500
    annual_num_workers : int, optional
        Number of workers to be used for parallel runs of the annual run, by default 24
    annual_noptmax : int, optional
        Number of iterations for the annual simulations by default 30
    monthly_num_workers : int, optional
        Number of number workers for the monthly simulations, by default 24
    monthly_noptmax : int, optional
        Number of iterations for the monthly simulations, by default 6
    use_focus_weights : bool, optional
        Assign a higher weight to more recent data, by default False
    use_obs_diff : bool, optional
        Use observed differences, by default True
    use_delay : bool, optional
        Include delay interbeds in the subsidence simulations, by default True
    plot : bool, optional
        Plot the results, by default True
    port : int, optional
        Defined port for PEST++ to use, by default None. If None, a random port number 
        between 4000 and 9000 is used.

    Returns
    -------
    None

    """
  
    if sites is None:
        sites = ["B88"] #Add the name of any site you want to run
    else:
        if isinstance(sites, str):
            sites = [sites]

    if port is None:
        port = random.randint(4000, 9000)

    for site_name in sites:
        prefer_less_rebound = [0.1, 0.1] #To constrain rebound, set this to numbers between 0.1 to 0.5
        ibtie = ["all", True]  #Allows you to run the model with properties tied across all layers or by layer
        quantiles = [[0.1, 0.25, 0.33, 0.66, 0.75, 0.9], [0.1, 0.25, 0.33, 0.66, 0.75, 0.9]]
        exp_tags = [f"_{date_string}", f"_{date_string}"]

        scenariofiles = ["CH_forecast", "DWR_scenarios", "2015_scenario"] #This allows you to run the scenarios

        for pref, ib, quants, exp_tag in zip(prefer_less_rebound, ibtie, quantiles, exp_tags):
            workflow.analyze_site(site_name, num_reals=num_reals, noptmax=monthly_noptmax, usecondor=True,
                                  num_workers=monthly_num_workers, run=True, use_obs_diff=use_obs_diff,
                                  prep=True, plot=plot, use_delay=use_delay, use_focus_weights=use_focus_weights,
                                  experiment_tag=exp_tag,
                                  port=port, freq="M", include_ghb_pars=True,
                                  estimate_clay_thickness=False, run_scenarios=False, plot_scens=False,
                                  scenario_tag="",
                                  run_prior_scenarios=False, export_scenario_basereals=False,
                                  scenario_subset=scenario_subset,
                                  morris_scenario_name=None, specified_initial_interbed_state=True,
                                  assimilate_all=True, prefer_less_rebound=pref,
                                  prefer_less_delayedfuture=False, scenariofiles=scenariofiles, head_based=False,
                                  prepfunc_kwargs={"quantiles": quants},
                                  tie_ib_by_layer=ib, run_mod_fxn=False)

    workflow.gather_pdfs(fig_sites, combine_to_one=False)
    workflow.gather_csvs(fig_sites, combine_to_one=False)
    workflow.gather_pngs(fig_sites, combine_to_one=False)
    workflow.gather_results(fig_sites)

if __name__ == "__main__":

    main()


