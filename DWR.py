
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
        annual_num_workers= 48,
        annual_noptmax= 30,
        monthly_num_workers= 48,
        monthly_noptmax=6, 
        use_focus_weights=False,
        use_obs_diff=True,
        use_delay=True,
        plot=True,
        port=None,
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
        sites = ["T88", "212.8"]
    else:
        if isinstance(sites, str):
            sites = [sites]

    if port is None:
        port = random.randint(4000, 9000)

    for site_name in sites:
        exp_tag = f"_{date_string}"
        prefer_less_rebound = 1e-20
        if site_name in [""]: #If you need less rebound for a specific site include it here
            prefer_less_rebound = 0.4
            rb_tag = str(prefer_less_rebound).split(".")[1]
            exp_tag = f"_{date_string}_{rb_tag}"

        if site_name in ['GWM_14', 'T88']:
            num_reals = 250
            annual_noptmax = 30
            monthly_noptmax = 25

        m_d = workflow.analyze_site(site_name, num_reals=num_reals, noptmax=annual_noptmax,
                                    num_workers=annual_num_workers, run=True, use_obs_diff=use_obs_diff,
                                    prep=True, plot=plot, use_delay=use_delay, use_focus_weights=use_focus_weights,
                                    experiment_tag=exp_tag,
                                    port=port, freq="Y", include_ghb_pars=True,
                                    estimate_clay_thickness=True, specified_initial_interbed_state=False,
                                    prefer_less_rebound=prefer_less_rebound)

        workflow.analyze_site(site_name, num_reals=num_reals, noptmax=monthly_noptmax,
                              num_workers=monthly_num_workers, run=True, use_obs_diff=use_obs_diff,
                              prep=True, plot=plot, use_delay=use_delay, use_focus_weights=use_focus_weights,
                              experiment_tag=exp_tag,
                              port=port, freq="M", include_ghb_pars=True,
                              estimate_clay_thickness=True, specified_initial_interbed_state=False,
                              xfer_from_m_d=m_d, prefer_less_rebound=prefer_less_rebound)

    workflow.gather_pdfs(sites, combine_to_one=False)
    workflow.gather_csvs(sites, combine_to_one=False)
    workflow.gather_pngs(sites, combine_to_one=False)

if __name__ == "__main__":

    main()


