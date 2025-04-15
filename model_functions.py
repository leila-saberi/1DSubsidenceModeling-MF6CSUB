import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
from importlib import reload
import os
# os.chdir(r'C:\Projects\1D_CSUB_temp1')
sys.path.insert(0,os.path.abspath(os.path.join('dependencies')))
import numpy as np
import pandas as pd
import geopandas as gpd
import flopy
import pyemu
import matplotlib.pyplot as plt
import matplotlib as mpl
import os,datetime
import shutil 
from flopy.plot.styles import styles
import statistics 


"""
To Do:

- split up corcoran into 3 different layers
- add option to vertically split lower aquifer

"""

if sys.platform.startswith('win'):
    assert "dependencies" in pyemu.__file__
    assert "dependencies" in flopy.__file__
    bin_path = os.path.join("bin","win")
    bin_name = "mf6.exe"
    
elif sys.platform.startswith('linux'):
    assert "dependencies" in pyemu.__file__
    assert "dependencies" in flopy.__file__
    bin_path = os.path.join("bin","linux")
    bin_name = "mf6"

elif sys.platform.lower().startswith('dar') or sys.platform.lower().startswith('mac'):
    assert "dependencies" in pyemu.__file__
    assert "dependencies" in flopy.__file__
    bin_path = os.path.join("bin","mac")
    bin_name = "mf6"
else:
    raise Exception('***ERROR: OPERATING SYSTEM UNKOWN***')

def initialize_model(location,freq="Y",start_datetime=None,wl_sample="mean",
                     specified_initial_interbed_state=True):
    w_d = os.path.join(location)
    prop_df = pd.read_csv(os.path.join(w_d,"processed_data","{0}.model_property_data.csv".format(location)),index_col=0)
    org_gwelev_df = pd.read_csv(os.path.join(w_d,'processed_data','{0}.ts_data.csv'.format(location)),index_col=0, parse_dates=True)
    sub_df = pd.read_csv(os.path.join(w_d,"source_data","{0}_sub_data.csv".format(location)),index_col=0,parse_dates=True)

    prop_df.columns = [int(c) for c in prop_df.columns]
    assert prop_df.dropna().shape == prop_df.shape
    
    for idx in prop_df.index:
        try:
            prop_df.loc[idx,:] = prop_df.loc[idx,:].astype(float)
        except Exception as e:
            pass
    gwelev_dfs = {}
    kvals = org_gwelev_df.klayer.unique()
    kvals.sort()
    udts = set()
    start_passed = True
    if start_datetime is None:
        start_datetime = min(sub_df.index.min(),org_gwelev_df.index.min())
        start_passed = False
    if isinstance(start_datetime,str):
        start_datetime = pd.to_datetime(start_datetime)
    udts.update(set([start_datetime - pd.to_timedelta(1,unit='d')]))
    # print(start_datetime)
    # exit()
    for kval in kvals:
        kdf = org_gwelev_df.loc[org_gwelev_df.klayer==kval,["interpolated"]].copy()
        kdf.sort_index(inplace=True)
        historic_dts = pd.date_range(start_datetime,kdf.index.max(),freq='d')
        kdf = kdf.reindex(historic_dts,method="nearest")

        if start_passed:
            kdf = kdf.loc[kdf.index>=start_datetime]
        if wl_sample.strip().lower() == "mean":
            kdf_resample = kdf.resample(freq).mean().interpolate(method='time').ffill().bfill()
        elif wl_sample.strip().lower() == "min":
            kdf_resample = kdf.resample(freq).min().interpolate(method='time').ffill().bfill()
        else:
            raise Exception("unsupported 'wl_sample' arg: '{0}'".format(wl_sample))
        gwelev_dfs[kval] = kdf_resample
        udts.update(set(kdf_resample.index.tolist()))
    historic_dts = list(udts)
    historic_dts.sort()
    
    historic_dts = pd.DatetimeIndex(historic_dts)
    historic_perlen = list((historic_dts[1:] - historic_dts[:-1]).days.values)
    historic_perlen.insert(0,1)

    nlay = prop_df.columns.max() + 1

    # Define the spatial discretization
    nr,nc = 1,1
    delr = delc = 1.
    top = prop_df.loc["top",:].max() 
    botm = prop_df.loc["botm",:].values
    nstp = 1
    pred_end = pd.to_datetime("12-31-2060")
    pred_drange = pd.date_range(historic_dts.max(),pred_end,freq=freq)
    perlen_pred = (pred_drange[1:] - pred_drange[:-1]).days.values
    perioddata = []
    ghbdata = {}
    steady_state = {}
    transient = {}
    
    #sim_start_datetime = (historic_dts.min() - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    #first_delta = (historic_dts[1] - historic_dts[0]).days + 1
    #sim_start_datetime = (historic_dts.min() - pd.DateOffset(days=first_delta)).strftime('%Y-%m-%d')
    #perioddata.append((1,1,1)) # Start with a year long period for steady state initial conditions
    sim_start_datetime = historic_dts.min().strftime('%Y-%m-%d')
    steady_state[0] = True
    transient[1] = True
    # Initialize CHD with the first time step
    cond = 50000
    ghbspdata = []
    last_dict = {}
    #for k,h in zip(gwelev_df.columns.values,gwelev_df.iloc[0,:].values):
    for k in range(nlay):
        if k not in gwelev_dfs:
            continue
        df = gwelev_dfs[k]
        h = df.interpolated.iloc[0]
        ghbspdata.append(((k,0,0),h, cond))
        last_dict[k] = (0, ghbspdata[-1])
        print(k,h)
 
    ghbdata[0] = ghbspdata
    ghb_warnings = []
    tol = 0.1 #hard coded in ies functions also
    for kper,(dt,perlen) in enumerate(zip(historic_dts,historic_perlen)):
        ghbspdata = []
        perioddata.append((perlen,nstp,1))
        for k in range(nlay):
            if k not in gwelev_dfs:
                continue
            df = gwelev_dfs[k]
            if dt not in df.index:
                ghbspdata.append(last_dict[k][1])
                print("WARNING: filling missing ghb data for kper {0} and layer {1}, dt:{2}".format(kper+1,k+1,str(dt)))
                ghb_warnings.append("WARNING: filling missing ghb data for kper {0} and layer {1}, dt:{2}".format(kper+1,k+1,str(dt)))

                continue
            h = df.loc[dt,"interpolated"]
            if h <= (botm[k]+tol):
                print("WARNING: ghb level {0} below cell bottom {1} for layer {4}, kper {2}, dt:{3}".format(h,botm[k]+tol,kper,str(dt),k+1))
                ghb_warnings.append("WARNING: ghb level {0} below cell bottom {1} for layer {4}, kper {2}, dt{3}".format(h,botm[k]+tol,kper,str(dt),k+1))
                continue
                #h = botm[k]+tol
                
            ghbspdata.append(((k,0,0),h, cond))
            last_dict[k] = (kper+1,ghbspdata[-1])
        ghbdata[kper+1] = ghbspdata

    #perioddata.append((3650.*3,nstp,1))
    if len(ghb_warnings) > 0:
        with open(os.path.join(w_d,"processed_data","ghb_warnings.dat"),'w') as f:
            for w in ghb_warnings:
                f.write(w+"\n")


    kvals = list(last_dict.keys())
    kvals.sort()
    #ghbdata[kper+1] = [last_dict[k][1] for k in kvals]

    kper += 1
    for kper_pred,perlen in enumerate(perlen_pred):
        perioddata.append((perlen,nstp,1))
        ghbdata[kper+kper_pred] = [last_dict[k][1] for k in kvals]
    print(len(ghbdata), kper)
    assert len(perioddata) == kper + len(perlen_pred)
    assert len(ghbdata) == len(perioddata)
    # Solver parameters
    nouter = 300
    ninner = 200
    hclose = 1e-3
    rclose = 1e-3
    linaccel = "bicgstab"
    relax = 0.97
    
    
    ndelaycells = 19
    ninterbeds = nlay
    
    
    update_material_properties = False
    #if "delay" not in prop_df.loc["cdelay",:].str.lower():
    #    update_material_properties = False
    sub6 = []
    for k in range(nlay):
        ['icsubno','k','ic','jc','cdelay','pcs0','thick_frac','rnb','ssv_cc','sse_cr','theta','ib_kv','h0']    
        ksub6 = [k,(k,0,0)]
        ksub6.append(prop_df.loc['cdelay',k])
        ksub6.append(prop_df.loc['pcs0',k])
        ksub6.append(prop_df.loc['thick_frac',k])
        ksub6.append(prop_df.loc['rnb',k])
        ksub6.append(prop_df.loc["ssv_cc",k])
        ksub6.append(prop_df.loc["sse_cr",k])
        ksub6.append(prop_df.loc["theta",k])
        ksub6.append(prop_df.loc["kv",k])
        ksub6.append(prop_df.loc["h0",k])
        
        sub6.append(ksub6)
    
    # Set up the model
    modelname = "model"#f"{location}.{scenario}" if scenario is not None else f"{location}.historical"
    model_ws = os.path.join(location,'model_ws','historical')
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)
    os.makedirs(model_ws)
    shutil.copy2(os.path.join(bin_path,bin_name),os.path.join(model_ws,bin_name))
   
       
    exe_name = 'mf6'
    sim = flopy.mf6.MFSimulation(sim_name=modelname, version='mf6', exe_name=exe_name, sim_ws=model_ws,continue_=True)
    tdis = flopy.mf6.ModflowTdis(sim, nper=len(perioddata),time_units='DAYS', perioddata=perioddata,
                                 start_date_time=sim_start_datetime)
    
    # Set IMS solver of flow model to use Newton-Ralphson solution
    ims = flopy.mf6.ModflowIms(sim, 
                         print_option="summary",
                         complexity="simple",
                         outer_maximum=nouter,
                         outer_dvclose=hclose,
                         inner_dvclose=rclose,
                         linear_acceleration=linaccel,
                         inner_maximum=ninner,
                         relaxation_factor=relax)
    # Create the groundwater flow model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=modelname, save_flows=True,
                               newtonoptions="newton")

    

    # Create the discretization package
    dis = flopy.mf6.ModflowGwfdis(gwf, length_units='FEET', nlay=nlay, nrow=nr, ncol=nc, 
                                  delr=delr, delc=delc, top=top, botm=botm)

    # Create the initial conditions package
    # since kper 0 is ss, these shouldnt matter...
    ic = flopy.mf6.ModflowGwfic(gwf, strt=org_gwelev_df.interpolated.max())

    # Create the node property flow package
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True, icelltype=1, 
                                  k=prop_df.loc["k",:].values,
                                  k33=prop_df.loc["k33",:].values)

    # Create the storage package
    sto = flopy.mf6.ModflowGwfsto(gwf, save_flows=True, iconvert=0, ss=0, sy=0, steady_state=steady_state,
                                  transient=transient)

    # Create the general head boundary package
    ghbkper = list(ghbdata.keys())
    ghbkper.sort()
    for iper in ghbkper:
        if len(ghbdata[iper]) == 0:
            print("empty ghb info for stress period ",iper+1)
    ghb = flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghbdata, save_flows=True)

    # Create the subsidence package
    csub = flopy.mf6.ModflowGwfcsub(gwf, print_input=True,save_flows=True,
                                    head_based=True,
                                    initial_preconsolidation_head=False,
                                    update_material_properties=update_material_properties,
                                    specified_initial_interbed_state=specified_initial_interbed_state,
                                    ndelaycells=ndelaycells,
                                    beta= 2.2270e-8,
                                    gammaw=62.48,
                                    ninterbeds=ninterbeds,
                                    sgm=prop_df.loc["sgm",:].values,
                                    sgs=prop_df.loc["sgm",:].values,
                                    cg_theta=prop_df.loc["cg_theta",:].values,
                                    cg_ske_cr=prop_df.loc["cg_ske_cr",:].values,
                                    packagedata=sub6,
                                    zdisplacement_filerecord=f"{modelname}.displacement.hds",
                                    package_convergence_filerecord=f"{modelname}.conv.log",)
    
    
    opth = f"{modelname}.gwf.obs"
    cpth = opth + ".csv"
    obs_array = []
    for k in range(nlay):
        obs_array.append(
            [
                f"HD.{k + 1:02d}",
                "HEAD",
                (k, 0, 0),
            ]
        )
    flopy.mf6.ModflowUtlobs(gwf,digits=10,print_input=True,filename=opth,continuous={cpth: obs_array})
    # Create obs package for every layer of total cell compaction
    
    opth = f"{modelname}.csub.obs"
    csub_csv = opth + ".csv"
    obslist = []
    for obsvalue in ['compaction','preconstress','elastic-compaction','inelastic-compaction']:
        for k in range(nlay):
            tag = f"{obsvalue}.{k + 1:02d}"
            obslist.append(
                            (tag,f"{obsvalue}-cell",(k, 0, 0))
                        )
    for obsvalue in ['delay-preconstress','delay-head']:
        for k in range(1,nlay):
            tag = f"{obsvalue}.{k + 1:02d}"
            obslist.append(
                                (tag+".{0}".format(int(ndelaycells)-1),
                                    f"{obsvalue}",(k, 0, int(ndelaycells)-1))
                            )
            # for i in range(ndelaycells):
            #     obslist.append(
            #                     (tag+".{0}".format(i+1),f"{obsvalue}",(k, 0, i))
            #                 )
    orecarray = {csub_csv: obslist}
    csub.obs.initialize(
        filename=opth, digits=10, print_input=True, continuous=orecarray
    )
    
    oc = flopy.mf6.ModflowGwfoc(gwf, printrecord=[("BUDGET", "ALL")], budget_filerecord=f"{modelname}.cbc",
                                saverecord=[("HEAD", "ALL"),("BUDGET", "ALL")], head_filerecord=f"{modelname}.hds",
                                budgetcsv_filerecord="budget.csv")
    
    sim.set_all_data_external()
    sim.write_simulation()
    
    return sim, model_ws

def clean_model(location):
    modelname = "model"
    model_ws = os.path.join(location,'model_ws','historical')
    
    
    # Clean up weird name in csub obs output file
    opth = f"{modelname}.csub.obs"
    csub_input = opth + "input.txt"
    f = open(os.path.join(model_ws,opth),'r')
    lines = f.readlines()
    f.close()
    f = open(os.path.join(model_ws,opth),'w')
    for line in lines:
        if 'OPEN/CLOSE' in line:
            oname = line.split("'")[1].split("'")[0]
            f.write(f'    OPEN/CLOSE  {csub_input}\n')
        else:
            f.write(line)
    f.close()
    
    os.replace(os.path.join(model_ws,oname),os.path.join(model_ws,csub_input))
    
    

def build_model(location,use_delay,prepfunc_kwargs={},prerun=True,
                freq='Y',start_datetime=None,wl_sample="mean",
                specified_initial_interbed_state=True):
    
    sys.path.insert(0,location)
    import prep_data
    reload(prep_data)
    print(prep_data.__file__)
    print(location)
    assert location in prep_data.__file__

    # super hack...    
    b_d = os.getcwd()
    os.chdir(location)
    # if rolling_window is not None:
    #     prep_data.prep_data(use_delay,rolling_window)
    # else:
    prep_data.prep_data(use_delay,**prepfunc_kwargs)
    os.chdir(b_d)
    

    sim,model_ws = initialize_model(location,freq=freq,start_datetime=start_datetime,
                                    wl_sample=wl_sample,
                                    specified_initial_interbed_state=specified_initial_interbed_state)
    clean_model(location)
    
    if prerun:
        pyemu.os_utils.run("mf6",cwd=model_ws)
        
def run_all(locations=None, freq=None):
    
    d = "."
    if locations is None:
        locations = [f for f in os.listdir(d) if f not in ['THmodel','bin','CSUB_example','CSUB_EffectiveStressFormulation'] and "." not in f]
    for location in locations:
        print('Working on:',location)
        scenarios = pd.ExcelFile(os.path.join(location,'source_data',
                                               f"{location}_scenario_data.xlsx")).sheet_names
        scenarios.sort(reverse=True)
        scenarios.append(None)
        for scenario in scenarios:
            if location not in scenario:
                print('Building and running:',location,scenario)
                build_model(location,freq=freq,
                            scenario=scenario,prerun=True)

def plot_results(location):
        
    fig,ax = plt.subplots(1,1,figsize=(10,6))
    scenarios = [f for f in os.listdir(os.path.join(location,'model_ws')) if not f.endswith('.png')]
    model_ws = os.path.join(location,'model_ws','historical') 
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws)
    nlay = sim.get_model().dis.nlay.data
    # get model
    model = sim.get_model()
    for scenario in scenarios:
        modelname = "model"#f"{location}.{scenario}"
        model_ws = os.path.join(location,'model_ws',scenario) 
        sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws)
        start_date_time = pd.to_datetime(sim.tdis.start_date_time.data)
        
        
        head = pd.read_csv(os.path.join(model_ws,"model.gwf.obs.csv"),index_col=0)
        sub = pd.read_csv(os.path.join(model_ws,"model.csub.obs.csv"),index_col=0)
        sub = sub.loc[:,sub.columns.str.contains('COMPACTION')].copy()
        sub['total'] = sub.iloc[:,1:].sum(axis=1)
        subobs = pd.read_csv(os.path.join(location,'source_data',f'{location}_sub_data.csv'),index_col=0, parse_dates=True)
        
        dates = [start_date_time + pd.Timedelta(days=i) for i in head.index]
        dates = pd.DatetimeIndex(dates)
        
        if scenario=="historical":
            ax.plot(dates,head.values,label=f'{col}-historical')  
        else:
            for col in head.columns:
                
                ax.plot(dates[dates>"12/31/2023"],
                        head[col].values[dates>"12/31/2023"],
                        label=f'{col}-{scenario}', linestyle="--")
        ax.legend(loc='lower left')
        ax.set_title(f'{location} {scenario}')
        ax.set_ylabel('Head (ft)')
        ax.grid(which='both',axis='both')
        sax = ax.twinx()
        sax.plot(dates,sub.total.values,label='simulated compaction',
                color='k',linestyle='--', linewidth=2)
        subobs.Subsidence_ft.plot(ax=sax,label='observed compaction',
                                    color='r',linestyle='--', 
                                    marker='o', linewidth=2, markersize=5)
        sax.legend(loc='upper right')
        sax.set_ylabel('subsidence (ft)')
        sax.invert_yaxis()
        
        fig.savefig(os.path.join(model_ws,'..',f"{location}.png"))
        
def plot_results_scenario(location,scenario):
    if scenario is  None:
        scenario = 'historical'
    modelname = "model"#f"{location}.{scenario}" 
    model_ws = os.path.join(location,'model_ws',scenario) 
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws)
    start_date_time = pd.to_datetime(sim.tdis.start_date_time.data)
    

    head = pd.read_csv(os.path.join(model_ws,"model.gwf.obs.csv"),index_col=0)
    sub = pd.read_csv(os.path.join(model_ws,"model.csub.obs.csv"),index_col=0)
    sub = sub.loc[:,sub.columns.str.contains('COMPACTION')].copy()
    sub['total'] = sub.iloc[:,1:].sum(axis=1)
    subobs = pd.read_csv(os.path.join(location,'source_data',f'{location}_sub_data.csv'),index_col=0, parse_dates=True)

    dates = [start_date_time + pd.Timedelta(days=i) for i in head.index]
    fig,ax = plt.subplots(1,1,figsize=(10,6))
    for col in head.columns:
        ax.plot(dates,head[col].values,label=col)
    ax.legend(loc='lower left')
    ax.set_title(f'{location} {scenario}')
    ax.set_ylabel('Head (ft)')
    ax.grid(which='both',axis='both')
    sax = ax.twinx()
    sax.plot(dates,sub.total.values,label='simulated compaction',
             color='k',linestyle='--', linewidth=2)
    subobs.Subsidence_ft.plot(ax=sax,label='observed compaction',
                                color='r',linestyle='--', 
                                marker='o', linewidth=2, markersize=5)
    sax.legend(loc='upper right')
    sax.set_ylabel('subsidence (ft)')
    sax.invert_yaxis()
    
    fig.savefig(os.path.join(model_ws,"results.pdf"))


if __name__ == "__main__":
    #d = os.path.join("C:\Projects","1D_CSUB")
    build_model('376.676',use_delay=True,prerun=True)
    #initialize_model("OCTOL","MS")
    #build_model('J88',freq='1Y',scenario="baseline",prerun=True)
    #run_all(locations=['J88'], freq='1Y')
    #plot_results_scenario('J88',None)
    
    #run_all()
