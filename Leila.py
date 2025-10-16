import workflow
import ies_functions
import pandas as pd
import os
import sys
from importlib import reload
from datetime import date
import random

today = date.today()
date_string = today.strftime("%m%d%Y")

def run_window_experiments():
    sites = ['D454']
    num_workers = 20
    num_reals = 200
    noptmax = 7
    use_focus_weights = True
    use_obs_diff = True
    run = True
    plot = True
    prep = True

    # jellis says
    start_datetime = pd.to_datetime("1-apr-2024")
    early_datetime = pd.to_datetime("1-1-1890")
    window_start_datetimes = [early_datetime]
    for window in [5, 10, 20]:
        window_start_datetime = start_datetime - pd.to_timedelta(window * 365, unit="d")
        window_start_datetimes.append(window_start_datetime)
    print(window_start_datetimes)
    window_labels = ["fullwindow", "_5yrwindow", "_10yrwindow", "_20yrwindow"]
    for label, start_datetime in zip(window_labels, window_start_datetimes):

        for site_name in sites:
            workflow.analyze_site(site_name, num_reals=num_reals, noptmax=noptmax,
                                  num_workers=num_workers, run=run, use_obs_diff=use_obs_diff,
                                  prep=prep, plot=plot, use_delay=True, use_focus_weights=use_focus_weights,
                                  experiment_tag="annual" + label, estimate_clay_thickness=True,
                                  port=4005, freq='Y', include_ghb_pars=True,
                                  verification_window=[early_datetime, start_datetime])

    # workflow.gather_pdfs()
    workflow.gather_results(sites, tag="window")
    r_d = "resultswindow"
    ies_functions.plot_window_compare(r_d, tag="window")

def main():
    annual_num_workers = 120#10 #24
    num_reals = 500 #5 #250
    annual_noptmax = 50 #2 #30
    monthly_num_workers = 120 #10 #24
    monthly_noptmax = 35 #2 #25
    use_focus_weights = True
    use_obs_diff = True
    use_delay = True
    scenario_subset = 100
    pcs0_range = 100

    port = random.randint(4000, 9000)

    # sites = ["341.804", "T200", "199.022",
    #           "OCTOL", "Y88", "438.939", "Q288", "212.8",
    #           "U822", "233.320", "250", "376.676", "AQUEDUCT", "D_289", "K67", "PTS2", "T54_RESET", "T949R", "F286"]

    sites = [ "19AA", "199.022"]#, "D454","U822", "J88", "H201", "341.438", "GWM_14", "H201","OCTOL","T949R", "Q288", "EARLIMART", "D454", ]
    # sites = ["PORT"]
    #rebound_dict = {"341.804":0.2,"H201":0.2,"EARLIMART":0.5}  # "PORT":0.5}#"341.804":0.2,"H201":0.2,"EARLIMART":0.5}

    for site_name in sites:
        # first run the no-specified state option
        # exp_tag = f"_{date_string}"
        prefer_less_rebound = [0.1, 0.1]  # ,.1,.1]
        ibtie = ["all", True]  # ,False,True]
        quantiles = [[0.1, 0.25, 0.33, 0.66, 0.75, 0.9], [0.1, 0.25, 0.33, 0.66, 0.75, 0.9]]
        exp_tags = [f"_{date_string}", f"_{date_string}"]

        if site_name in ["D454", "EARLIMART", "T88", "U822"]:
            scenariofiles = ["TH_reduction_scenario", "CH_forecast", "DWR_scenarios", "2015_scenario"]
        elif site_name == "EARLIMART":
            scenariofiles = ["DWR_scenarios", "CH_forecast", "2015_scenario", "MTNoMO_scenario"]
        elif site_name == "376.676":
            scenariofiles = ["DWR_scenarios", "CH_forecast", "2015_scenario", "HCMT_scenario"]
        else:
            scenariofiles = ["CH_forecast"] #, "2015_scenario", "DWR_scenarios", ]
        assert [s in ["DWR_scenarios", "CH_forecast", "2015_scenario", "MTNoMO_scenario", "HCMT_scenario"] for s in
                scenariofiles]

        scenariofiles = ["CH_forecast", "DWR_scenarios", "2015_scenario"]
        for pref, ib, quants, exp_tag in zip(prefer_less_rebound, ibtie, quantiles, exp_tags):
            # m_d = workflow.analyze_site(site_name,num_reals=num_reals,noptmax=annual_noptmax,usecondor = True,
            #         num_workers=annual_num_workers,run=True,use_obs_diff=use_obs_diff,
            #         prep=port,plot=True,use_delay=use_delay,use_focus_weights=use_focus_weights,
            #         experiment_tag=exp_tag,
            #         port=4005,freq="Y",include_ghb_pars=True,
            #         estimate_clay_thickness=False,run_scenarios=False,plot_scens=False,scenario_tag="",
            #         run_prior_scenarios=False,export_scenario_basereals=False,scenario_subset=scenario_subset,
            #         morris_scenario_name=None,specified_initial_interbed_state=True,
            #         assimilate_all=True,prefer_less_rebound=pref,
            #         prefer_less_delayedfuture=False,scenariofiles=scenariofiles,head_based=False,
            #         prepfunc_kwargs={"quantiles":quants},
            #         tie_ib_by_layer=ib,run_mod_fxn=False)
            m_d = workflow.analyze_site(site_name, num_reals=num_reals, noptmax=monthly_noptmax, usecondor = True,
                                        num_workers=monthly_num_workers, run=True, use_obs_diff=use_obs_diff,
                                        prep=True, plot=True, use_delay=use_delay, use_focus_weights=use_focus_weights,
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

    fig_sites = sites
    workflow.gather_pdfs(fig_sites, combine_to_one=False)
    workflow.gather_csvs(fig_sites, combine_to_one=False)
    workflow.gather_pngs(fig_sites, combine_to_one=False)
    workflow.gather_results(fig_sites)

if __name__ == "__main__":
    # main()
    # print("test")
    main()

    # ies_functions.plot_WL_SUB_2_panel(m_d=os.path.join('PORT', 'master_ies_delay_y_nofocus_nospecstate_lessrbnd_02172025_40pRB'))
    # workflow.prep_CH_scen_Input("PORT")

    # workflow.gather_pdfs(sites=['D454', 'T88', 'EARLIMART'])
    # workflow.gather_csvs(sites=['D454', 'T88', 'EARLIMART'])

    # ies_functions.plot_en_compaction(os.path.join("341.804", "master_ies_delay_focusmonthly_scenarios"))

