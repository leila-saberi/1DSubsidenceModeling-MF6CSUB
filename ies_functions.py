# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:53:09 2024

@author: aestrada
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os, sys
from importlib import reload
# os.chdir(r'C:\Projects\1D_CSUB_temp')
sys.path.insert(0,os.path.abspath(os.path.join('dependencies')))
import shutil
import numpy as np
import pandas as pd
from scipy import stats
# print('pandas version' + pd.__version__)
import pyemu
import flopy
import platform
import model_functions
import glob
import re

if sys.platform.startswith('win'):
    bin_path = os.path.join("bin","win")
    
elif sys.platform.startswith('linux'):
    bin_path = os.path.join("bin","linux")


elif sys.platform.lower().startswith('dar') or sys.platform.lower().startswith('mac'):
    bin_path = os.path.join("bin","mac")
    
else:
    raise Exception('***ERROR: OPERATING SYSTEM UNKOWN***')


# More concise labels for the plots for each scenario # todo fill missing labels
label_keys = {'mean-ch': 'CH Level',
              'ch-+20-ft': 'CH + 20ft',
              'ch-+50-ft': 'CH + 50ft',
              # '2015': '2015 WL',
              '2015-wls': '2015 WL',
              'mt-glidepath-not-below-mt': 'MO/MT',
              'mt-(glidepath-not-below-mt)': 'MO/MT',
              'injection-rebound-@2020-23-rate': 'injection-rebound-@2020-23-rate',
              'historical-lows': 'historical-lows',
              'no-sgma': 'no-sgma',
              'mtnomo': 'MT',
              'base':'base',
              'tcwa-reduced-10pct': 'TCWA 10p Reduction',
              'tcwa-reduced-20pct': 'TCWA 20p Reduction',
              'tcwa-reduced-30pct': 'TCWA 30p Reduction',
              'tcwa-reduced-50pct': 'TCWA 50p Reduction',
              'pixley-reduced-10pct': 'Pixley 10p Reduction',
              'pixley-reduced-20pct': 'Pixley 20p Reduction',
              'pixley-reduced-30pct': 'Pixley 30p Reduction',
              'pixley-reduced-50pct': 'Pixley 50p Reduction',

              }


def prep_for_parallel(b_d, ct_d, noptmax=-1,
                      num_reals=None, overdue_giveup_fac=10.0,
                      overdue_giveup_time=10.0):
    """prepare a given template directory for parallel execution.  Makes a directory copy with
    temporary files removed, and (optionally) a master directory

    Args:
        b_d (str): base template directory.  This is copied to b_d+"_parallel"
        pst_name (str): control file name
        mod_name (str): pst control file prefix
        noptmax (int): number of PESTPP-IES DA iterations
        ies_num_reals (int): number of model realizations to use from ensemble
        overdue_giveup_fac (float): factor to use to limit the amount of time waiting for
            slow runs to finish.
        overdue_giveup_time (float): maximum length of time to allow a model realization to run, in minutes.
        restart (str): if m_d is not None, relative path location of previously completed master directory
        max_ies_reals (int): number of model realizations to include in ensemble
    """
    # b_d='model_template'
    # ies_num_reals=nreal
    # overdue_giveup_fac=1.5
    # overdue_giveup_time=2.0
    # restart=master_dir
    # max_ies_reals=nreal 
    
    
    print(f'copying directory {b_d} to {ct_d}')
    if os.path.exists(ct_d):
        shutil.rmtree(ct_d)
    shutil.copytree(b_d, ct_d)
    
    # remove any temp files not needed since the parallel template dir will
    # need to be copied a bunch of times
    print(f'removing unneeded files from {ct_d}')
    # remove files present after pest run that were not in template file before pest run
    if os.path.exists(os.path.join(ct_d,'output_files.txt')):
        with open(os.path.join(ct_d,'output_files.txt')) as ifp:
            rm_files = ifp.read().splitlines()
        rm_files = [x for x in rm_files if x in os.listdir(ct_d)]
        [os.remove(os.path.join(ct_d, rm_file)) for rm_file in rm_files]
        # remove output file pointer
        # os.remove(os.path.join(ct_d,'output_files.txt'))

    # remove other unneeded files (except for ".jcb" [prior.jcb])
    rm_ext_list = [".cbb", ".hds", ".log", ".lst", ".rec", ".rei", \
                    ".base.par", ".dis.grb", ".dbf", "head_obs.csv", "zbud.csv", ".shp", ".shx"]
    for rm_ext in rm_ext_list:
        rm_files = [f for f in os.listdir(ct_d) if f.lower().endswith(rm_ext)]
        [os.remove(os.path.join(ct_d, rm_file)) for rm_file in rm_files]

    
    pst = pyemu.Pst(os.path.join(ct_d, "pest.pst"))


    if num_reals is not None:
        # if the initial parameter ensemble is not in the ++ options,
        # set it to the prior ensemble
        
        pst.pestpp_options["ies_num_reals"] = num_reals

    
    # set overdue_giveup_fac
    pst.pestpp_options["overdue_giveup_fac"] = overdue_giveup_fac
    # set overdue_giveup_time
    pst.pestpp_options["overdue_giveup_minutes"] = overdue_giveup_time
    # specify number of threads for tpl & ins file processing
    #pst.pestpp_options["num_tpl_ins_threads"] = 4
    pst.pestpp_options["ies_bad_phi_sigma"] = 1.75
    if num_reals >= 1000:
        pst.pestpp_options["ies_multimodal_alpha"] = 0.15
        pst.pestpp_options["ies_num_threads"] = 4
    else:    
        pst.pestpp_options["ies_multimodal_alpha"] = 0.99
        pst.pestpp_options["ies_num_threads"] = 4
    if noptmax >= 10:
        pst.pestpp_options["ies_n_iter_mean"] = [-1,-2,9999]
    if noptmax >= 20:
        pst.pestpp_options["ies_n_iter_mean"] = [-1,-3,-3,9999] #testLessNoise
    if noptmax >= 30:
        pst.pestpp_options["ies_n_iter_mean"] = [-1,-3,-5,-5,9999] #testLessNoise
    if noptmax >= 40:
        pst.pestpp_options["ies_n_iter_mean"] = [-1,-3,-5,-5,-5,9999] #testLessNoise
        

    pst.control_data.nphinored = 10000
    pst.control_data.nphistp = 10000
    #pst.pestpp_options["panther_agent_freeze_on_fail"] = True

    # draw options
    pst.pestpp_options["ies_no_noise"] = False
    pst.pestpp_options["ies_group_draws"] = False

    #pst.pestpp_options["ies_initial_lambda"] = -10 # 10 * initial phi
    pst.pestpp_options["ies_subset_size"] = -15 # 10% of ensemble size used for evaluating Marquardt Lambdas

    #pst.pestpp_options["panther_agent_freeze_on_fail"] = True
    pst.pestpp_options["panther_transfer_on_fail"] = ["mfsim.lst","model.lst"]
    # number of iterations
    pst.control_data.noptmax = noptmax 
    
    # save the control file into the master dir
    print(f'writing control file in {ct_d}')
    pst.write(os.path.join(ct_d, "pest.pst"), version=2)



def run_local(worker_dir, master_dir, pst_name="pest.pst", num_workers=10, port=4004, pestpp="ies"):
    """run PESTPP-IES in parallel on the current machine

    Args:
        worker_dir (str): "base" directory that contains all the files needed
            to run PESTPP-IES (MODFLOW file and PEST interface files)
        master_dir (str): "master" directory that will be created and where the
            PESTPP-IES master instance will be started
        pst_name (str): control file name. Must exist in b_d
        num_workers (int): number of parallel workers to start.
            Default is 10.

    """    
    worker_root = os.path.split(worker_dir)[0]
    print("...worker root:",worker_root)
    pyemu.os_utils.start_workers(worker_dir, f"pestpp-{pestpp}", pst_name, num_workers=num_workers,
                                 master_dir=master_dir, worker_root=worker_root, reuse_master=False,
                                 port=port)





def prep_deps(d):
    """copy exes to a directory based on platform
    Args:
        d (str): directory to copy into
    Note:
        currently only mf6 for mac and win and mp7 for win are in the repo.
            need to update this...
    """
    # copy in deps and exes
    if "window" in platform.platform().lower():
        bd = os.path.join("bin", "win")
        
    elif "linux" in platform.platform().lower():
        bd = os.path.join("bin", "linux")
        
    else:
        bd = os.path.join("bin", "mac")
        
    for f in os.listdir(bd):
            shutil.copy2(os.path.join(bd, f), os.path.join(d, f))

    try:
        shutil.rmtree(os.path.join(d,"flopy"))
    except:
        pass

    shutil.copytree(os.path.join('dependencies', 'flopy'), os.path.join(d,"flopy"))

    try:
        shutil.rmtree(os.path.join(d,"pyemu"))
    except:
        pass

    shutil.copytree(os.path.join('dependencies',"pyemu"), os.path.join(d,"pyemu"))


def csub_pars_to_pkgdata(template_ws="."):
    df = pd.read_csv(os.path.join(template_ws,"csub_input.csv"),index_col=0)
    names = ["lay", "row", "col", "cdelay", "pcs0", "thick_frac", "rnb", "ssv_cc",
             "sse_cr", "theta", "kv", "h0"]
    df.index += 1 #to be icsubno
    if "clay_lith" in df.columns and "clay_thickness_squared" in df.columns and "clay_thickness" in df.columns:
        bequiv = np.sqrt(df["clay_thickness_squared"].values/df["clay_lith"].values)
        df["org_rnb"] = df.rnb.copy()
        df["org_thick_frac"] = df.thick_frac.copy()
        df["rnb"] = df['clay_thickness'] / bequiv
        rnb = df.rnb.values
        rnb[rnb<1.0] = 1.0
        df["rnb"] = rnb
        df["thick_frac"] = bequiv
        df["gwf-clay-thickness"] = df.thick_frac * df.rnb
        df["gwf-to-csub-ratio"] = df["gwf-clay-thickness"] / df["gwf_thickness"]
        df.loc[df["gwf-to-csub-ratio"]>1.0,"thick_frac"] = 0.99 * (df["thick_frac"] / df.loc[df["gwf-to-csub-ratio"]>1.0,"gwf-to-csub-ratio"])
        df["implied-clay-layer_thickness"] = df['clay_thickness'].values / df["clay_lith"].values
        df["implied-clay2-layer_thickness"] = np.sqrt(df['clay_thickness_squared'].values) / df["clay_lith"].values
        df["implied-thick-diff"] = df["implied-clay-layer_thickness"].values - df["implied-clay2-layer_thickness"]
        df["sse2ssv-ratio"] = df["sse_cr"] / df["ssv_cc"]
        print(bequiv)
        print(df["rnb"])
        print(df["thick_frac"])
        
        df.loc[df.cdelay.str.lower()=="nodelay","rnb"] = df.loc[df.cdelay.str.lower()=="nodelay","org_rnb"]
        df.loc[df.cdelay.str.lower()=="nodelay","thick_frac"] = df.loc[df.cdelay.str.lower()=="nodelay","org_thick_frac"]
        
    df.loc[:,names].to_csv(os.path.join(template_ws,"model.csub_packagedata.txt"),header=False,sep=" ")
    df.drop(["cdelay","row","col"],axis=1,inplace=True)
    df.index = df.pop("lay") - 1
    df.index.name = "k"
    df.to_csv(os.path.join(template_ws,"listinput_obs.csv"))

def replace_time_with_datetime(csv_file, add_subsidence=True, start_datetime='1904-12-31',
                               nlay=3): 
    start_datetime = pd.to_datetime(start_datetime)
    df = pd.read_csv(csv_file,index_col=0)
    df.columns = df.columns.str.lower()
    if 'budget' in csv_file:
        cols = []
        for c in df.columns:
            if len(c.split("(")) > 1:
                if not any(x in c for x in ['drn', 'riv', 'wel']):
                    cols.append(c.split("(")[0] + c.split(")")[1])
                else:
                    cols.append(c.split("(")[1].split(")")[0] + '_' + c.split('_')[-1])
            else:
                cols.append(c.lower())
        df.columns = cols                
    else:
        for col in df.columns:
            vals = df[col].values
            vals[vals>1e29] = -999
            df[col] = vals
    datetimes = start_datetime + pd.to_timedelta(df.index.values.astype(int),unit='d')
    dates = [x.strftime("%Y%m%d") for x in datetimes]
    df.loc[:, 'datetime'] = dates
    df.index = df.pop("datetime")
    raw = os.path.split(csv_file)
    new_file = os.path.join(raw[0],"datetime_" + raw[1])
    if add_subsidence:
        add_cols = [c for c in df.columns if 'compaction' in c and len(c)==13]
        df.loc[:, 'sim-subsidence-ft'] = df.loc[:, add_cols].sum(axis=1)
        df.to_csv(new_file) 
    else:
        df.to_csv(new_file) 
    return df



def get_input_obs(mod_name="model"):
    '''track csub parameter values'''
    par_data = pd.read_csv("mult2model_info.csv", index_col=0)
    #par_data.loc[:, 'pname'] = par_data.loc[:, 'org_file'].apply(lambda x: x[4:].split('_inst0_constant.csv')[0])
    par_data = par_data.loc[pd.isna(par_data.index_cols),:]
    par_data.loc[:, 'pname'] = par_data.loc[:, 'model_file'].apply(lambda x: x.split(".")[1])
    par_data = par_data.loc[~par_data.model_file.str.contains('stress_period')].reset_index(drop=True)
    par_data = par_data.loc[:, ['pname', 'upper_bound', 'lower_bound',"model_file"]]
    par_data.loc[:, 'pval'] = par_data.loc[:, 'model_file'].apply(lambda x: np.loadtxt(x))
    #par_data.loc[:, 'pname'] = par_data.loc[:, 'pname'].apply(lambda x: x.split('.')[2])
    par_data.to_csv('arrinput_obs.csv')
    return par_data
  

def initialize_input_obs(template_ws="template", mod_name="model"):
    b_d = os.getcwd()
    os.chdir(template_ws)
    df = get_input_obs(mod_name)
    os.chdir(b_d)
    return df


def setup_pst(org_d, site_name, template_ws=None, num_reals=100,include_ghb_pars=False,
    estimate_clay_thickness=False):
    
    mod_name = "model"#f'{site_name}.{version}'
    assert os.path.exists(org_d)
    
    # A dir to hold a copy of the org model files
    
    tmp_d = org_d + "_temp"
    
    if os.path.exists(tmp_d):
        shutil.rmtree(tmp_d)
    shutil.copytree(org_d, tmp_d)
    
    # Specify a template directory (i.e. the PstFrom working folder)
    if template_ws is None:
        template_ws = os.path.join(site_name, "template")
    
    # load simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d, load_only=['dis', 'tdis'])
    
    # load flow model
    gwf = sim.get_model()
    botm_dict = {k:gwf.dis.botm.array[k,0,0] for k in range(gwf.dis.nlay.data)}
    #print(botm_dict)
    
    # Model properties
    start_datetime = sim.tdis.start_date_time.data
    perlen = sim.tdis.perioddata.array["perlen"]
    use_monthly_ghbs = True
    corrlen = 365*1
    if perlen.mean() > 30:
        use_monthly_ghbs = False
        corrlen = 365*1
    nlay = gwf.dis.nlay.data
    botm = gwf.dis.botm.array
    thickness = {0:gwf.dis.top.array[0,0] - botm_dict[0]}
    for k in range(1,nlay):
        thickness[k] = botm_dict[k-1] - botm_dict[k]
   
    model_dts = pd.to_datetime(start_datetime) + pd.to_timedelta(np.cumsum(perlen),
                                                            unit='d')
    # instantiate PEST container
    pf = pyemu.utils.PstFrom(original_d=tmp_d, new_d=template_ws, remove_existing=True,
                              longnames=True, zero_based=False, echo=True,
                              start_datetime=start_datetime,
                             chunk_len=200)
    
    # intermediate files to remove during each forward run
    pf.tmp_files.append(f"{mod_name}.cbb")
    pf.tmp_files.append(f"{mod_name}.hds")
    pf.tmp_files.append(f"{mod_name}.lst")
    
    prep_deps(template_ws)

    cum,inc = get_lst_budget(template_ws,start_datetime=start_datetime)
    pf.add_py_function("ies_functions.py","get_lst_budget('.',start_datetime='{0}')".format(pd.to_datetime(start_datetime).strftime("%Y%m%d")),is_pre_cmd=False)
    pf.add_observations("cum.csv",index_cols=[cum.index.name],use_cols=cum.columns.tolist(),prefix="lstcum",obsgp="lstcum",ofile_sep=",")
    pf.add_observations("inc.csv",index_cols=[inc.index.name],use_cols=inc.columns.tolist(),prefix="lstinc",obsgp="lstinc",ofile_sep=",")
    pf.add_observations("nfail.csv",index_cols=["onme"],use_cols=["oval"],prefix="nfail",obsgp="nfail",ofile_sep=",")


    # produce arrays from csub packagedata
    #raw_csub_df = csub_arrs_from_pkgdata(template_ws, mod_name)
    
    # GHB head as "additive" parameters
    # Add temporal correlation to WL pars
    temporal_v = pyemu.geostats.ExpVario(contribution=1.0, a=corrlen, anisotropy=1.0,
                                         bearing=0.0, name="temporal_v")
    temporal_gs = pyemu.geostats.GeoStruct(variograms=temporal_v, transform='none',
                                           name="temporal_gs") 
    
    df = test_csv_to_ghb(template_ws)
    pf.add_py_function("ies_functions.py", "csv_to_ghb()", is_pre_cmd=True)
    pf.add_parameters("input_ghb.csv", index_cols=["datetime", "kper", "k"], use_cols="bhead", par_type="constant",
                          par_style="a", transform="none", initial_value=0, lower_bound=-40, pargp="consthead",
                          par_name_base="consthead",
                          upper_bound=40, mfile_sep=",")
    if include_ghb_pars:
        upper_bound = 50
        lower_bound = -50
        df = test_csv_to_ghb(template_ws)
        pf.add_parameters("input_ghb.csv", index_cols=["datetime", "kper", "k"], use_cols="bhead", par_type="grid",
                          par_style="a", transform="none", initial_value=0, lower_bound=lower_bound, pargp="ghbhead",
                          par_name_base="ghbhead",upper_bound=upper_bound, mfile_sep=",")

        # if use_monthly_ghbs:
        #     pf.add_parameters("input_ghb.csv", index_cols=["datetime", "kper", "k"], use_cols="bhead", par_type="grid",
        #                       par_style="a", transform="none", initial_value=0, lower_bound=-20, pargp="monthlyghbhead",
        #                       par_name_base="monthlyghbhead",
        #                       upper_bound=20, mfile_sep=",")

        # pf.add_parameters("input_ghb.csv", index_cols=["datetime", "kper", "k"], use_cols="bhead", par_type="grid",
        #                   par_style="a", transform="none", initial_value=0, lower_bound=-20, pargp="annualghbhead",
        #                   par_name_base="annualghbhead",
        #                   upper_bound=20, mfile_sep=",")

        
    pf.add_py_function("ies_functions.py","enforce_ghb_botm(nlay={0})".format(nlay),is_pre_cmd=True)

    # these are just for scenarios...they get fixed later
    pf.add_parameters("input_ghb.csv", index_cols=["datetime", "kper", "k"], use_cols="bhead", par_type="grid",
                      par_style="d", transform="none", pargp="directghbhead",
                      par_name_base="directghbhead",
                      mfile_sep=",")

    org_pkg_df = pd.read_csv(os.path.join(site_name,"processed_data","{0}.model_property_data.csv".format(site_name)),index_col=0)
    keep_names = ["cdelay", "pcs0", "thick_frac", "rnb", "ssv_cc",
             "sse_cr", "theta", "kv", "h0","clay_thickness_squared","clay_lith","clay_thickness"]
    par_names = ["pcs0", "thick_frac", "rnb", "ssv_cc",
             "sse_cr", "theta", "kv", "h0","clay_thickness_squared","clay_lith","clay_thickness"]
    if not estimate_clay_thickness:
        par_names = par_names[:-3]
        keep_names = keep_names[:-3]

    pkg_df = org_pkg_df.loc[keep_names,:].T

    pkg_df.index = [int(i) for i in pkg_df.index]
    pkg_df.index.name = "k"
    
    print(pkg_df.index)
    pkg_df["lay"] = pkg_df.index + 1
    pkg_df["row"] = 1
    pkg_df["col"] = 1

    pkg_df["gwf_thickness"] = pkg_df.lay.apply(lambda x: thickness[x-1])
    
    pkg_df.to_csv(os.path.join(pf.new_d,"csub_input.csv"))
    pf.add_parameters("csub_input.csv",index_cols=['k'],par_type="grid",par_style="d",
                      use_cols=par_names,pargp=["csub" for _ in par_names],
                      par_name_base=["csub" for _ in par_names])

    pnames = ['cg_ske_cr', 'cg_theta']
    
    for pname in pnames: 
        
        for k in range(nlay):            
            f = f'{mod_name}.csub_{pname}_layer{k+1}.txt'
            

            if "theta" in pname:
                pf.add_parameters(f, par_type='constant', upper_bound=1.3, lower_bound=0.7,
                                  ult_ubound=0.5, ult_lbound=0.05,
                                  par_name_base=f'{pname}_lyr{k+1}', pargp=pname)
            else:     
                pf.add_parameters(f, par_type='constant', upper_bound=20, lower_bound=0.05,
                                  par_name_base=f'{pname}_lyr{k+1}', pargp=pname)
                                  ##ult_ubound=par_data.uubnd[idx]*1.2, ult_lbound=par_data.ulbnd[idx]/1.2,
                
    # # Parameterization of k33 on an aquifer basis 
    k33_files = os.listdir(template_ws)
    k33_files = [f for f in k33_files if f.startswith("model.npf_k33") and f.endswith(".txt")]
    k33_files.sort(key = lambda x: int(x.split(".")[-2][13:]))
    assert len(k33_files) == gwf.dis.nlay.data
    for k33_file in k33_files:
        k = int(k33_file.split("layer")[1].split('.')[0]) - 1
        val = np.loadtxt(os.path.join(template_ws,k33_file))

        pf.add_parameters(k33_file,par_type="constant",upper_bound=10,lower_bound=0.1,
                          ult_ubound=val*100,ult_lbound=val*0.01,pargp="k33_k:{0}".format(k),
                          par_name_base="k33_k:{0}".format(k))
    

    # load csub array data into package data list
    #initialize_csub_arrs_to_pkgdata(template_ws, mod_name,nlay)
    # add processing scripts to pst forward run
    #pf.add_py_function("ies_functions.py", "csub_arrs_to_pkgdata(nlay={0})".format(nlay), is_pre_cmd=True)
    csub_pars_to_pkgdata(template_ws)
    pf.add_py_function("ies_functions.py","csub_pars_to_pkgdata()",is_pre_cmd=True)
    pf.add_py_function("ies_functions.py", "get_input_obs()", is_pre_cmd=False)
    # add call to PRE processing script to pst forward run
    #pf.pre_py_cmds.append(f'csub_arrs_to_pkgdata("{mod_name}", nlay={nlay})')
    # add model run command
    pf.mod_sys_cmds.append("mf6")
    # import necessary libraries in forward_run.py
    # pf.extra_py_imports.append('flopy') 
    # add call to processing script to pst forward run
    #pf.post_py_cmds.append(f"get_input_obs('{mod_name}')")
   
    # Subsidence observations
    csub_out = f'{mod_name}.csub.obs.csv'
    # add processing script to pst forward run
    pf.add_py_function("ies_functions.py", "replace_time_with_datetime()",
                        is_pre_cmd=None)
    
    gwf_out = f'{mod_name}.gwf.obs.csv'
    pf.post_py_cmds.append(f'replace_time_with_datetime("{gwf_out}", start_datetime="{start_datetime}", add_subsidence=False)')

    df = replace_time_with_datetime(os.path.join(template_ws, gwf_out),
                               start_datetime=start_datetime, add_subsidence=False)
    gwf_df = pf.add_observations("datetime_" + gwf_out,
                                  index_cols=["datetime"], use_cols=df.columns.tolist(),
                                  prefix="hds", ofile_sep=",")


    # add call to processing script to pst forward run
    pf.post_py_cmds.append(f'replace_time_with_datetime("{csub_out}", start_datetime="{start_datetime}", add_subsidence=True, nlay={nlay})')

    df = replace_time_with_datetime(os.path.join(template_ws, csub_out),
                               start_datetime=start_datetime, add_subsidence=True, nlay=nlay)
    csub_df = pf.add_observations("datetime_" + csub_out,
                                  index_cols=["datetime"], use_cols=df.columns.tolist(),
                                  prefix="sub", ofile_sep=",")
    # csub_df.loc[:,'year'] = csub_df.loc[:,'obsnme'].apply(lambda x: x.split(':')[-1][0:4])
    # csub_df.loc[:, "weight"] = 0.
    # csub_df.loc[:, "datetime"] = csub_df.loc[:,"obsnme"].apply(lambda x: x[-8:])
    # csub_df.loc[:, 'source'] = 'mf6'
    # csub_df = csub_df.loc[:, ['obsnme', 'obsval', 'obgnme', 'weight', 'datetime', 'source']]
    
    
    # track csub inputs
    pf.parfile_relations.to_csv(os.path.join(pf.new_d, "mult2model_info.csv"))
    df = initialize_input_obs(template_ws, mod_name)
    input_df = pf.add_observations('arrinput_obs.csv',
                                   index_cols=['pname'], use_cols=["lower_bound","upper_bound",'pval'],
                                   prefix='arrobs', obsgp="arrobs",ofile_sep=",")
    
    pf.tmp_files.append("listinput_obs.csv")
    df = pd.read_csv(os.path.join(template_ws,"listinput_obs.csv"),index_col=0)
    pf.add_observations("listinput_obs.csv",index_cols=["k"],
                        use_cols=df.columns.tolist(),
                        prefix="listobs",obsgp="listobs")



    #if include_ghb_pars:
    df = test_ghbhead_to_csv(template_ws)
    pf.add_py_function("ies_functions.py","ghbhead_to_csv()",is_pre_cmd=False)
    ghbdf = pf.add_observations("ghb.csv",index_cols=["k","datetime"],use_cols=["bhead"],prefix="ghbhead",obsgp="ghbhead",ofile_sep=",")
        
    
    pf.add_py_function("ies_functions.py","process_diff_obs()",is_pre_cmd=False)
    df = setup_diff_obs(template_ws)
    pf.add_observations("diff_datetime_model.csub.obs.csv",index_cols=["dt1","dt2"],use_cols=["diff"],
        prefix="subdiff",obsgp="subdiff",ofile_sep=",")

    pf.add_py_function("ies_functions.py","process_delay_obs()",is_pre_cmd=False)
    df = process_delay_obs(template_ws)
    pf.add_observations("delay_diffs.csv",index_cols=["datetime"],use_cols=df.columns.tolist(),
                        prefix="delaydif",obsgp=df.columns.tolist(),ofile_sep=",")


    pf.add_py_function("ies_functions.py","process_compaction_tdiff_obs()",is_pre_cmd=False)
    df = process_compaction_tdiff_obs(template_ws)
    pf.add_observations("compaction_tdiffs.csv",index_cols=["datetime"],use_cols=df.columns.tolist(),
                        prefix="compact-tdif",obsgp=df.columns.tolist(),ofile_sep=",")
    

    pf.add_py_function("ies_functions.py","process_percent_comp_obs()",is_pre_cmd=False)
    df = process_percent_comp_obs(template_ws)
    pf.add_observations("percent_compaction.csv",index_cols=["datetime"],use_cols=df.columns.tolist(),
                        prefix="percent-compact",obsgp=df.columns.tolist(),ofile_sep=",")
    
    
    pf.add_py_function("ies_functions.py","process_compaction_at_above_obs()",is_pre_cmd=False)
    df = process_compaction_at_above_obs(template_ws)
    pf.add_observations("compaction_atabove.csv",index_cols=["datetime"],use_cols=df.columns.tolist(),
                        prefix="atabove-compact",obsgp=df.columns.tolist(),ofile_sep=",")

    # BUILD PEST  
    pst = pf.build_pst('pest.pst', version=2)
    
    par = pst.parameter_data
    #keep_names = ["cdelay", "pcs0", "thick_frac", "rnb", "ssv_cc",
    #         "sse_cr", "theta", "kv", "h0","clay_thickness","clay_lith"]

    ppar = par.loc[par.pname == "csub",:]
    par.loc[ppar.parnme,"parlbnd"] = ppar.parval1 * 0.05
    par.loc[ppar.parnme,"parubnd"] = ppar.parval1 * 20.0
    par.loc[ppar.parnme,"pargp"] = ppar.apply(lambda x: x.usecol+"_"+x.k,axis=1)

    rpar = par.loc[par.parnme.str.contains("rnb"),:]
    par.loc[rpar.parnme,"parlbnd"] = 1.0

    if estimate_clay_thickness:
        ppar = par.loc[par.parnme.str.contains("rnb"),:]
        par.loc[ppar.parnme,"partrans"] = "fixed"
        ppar = par.loc[par.parnme.str.contains("thick_frac"),:]
        par.loc[ppar.parnme,"partrans"] = "fixed"
        # ppar = par.loc[par.parnme.str.contains("clay_thickness_pstyle"),:].copy()
        # print(ppar)
        # assert ppar.shape[0] == nlay
        # ppar["k"] = ppar.k.astype(int) + 1
        # par.loc[ppar.parnme.values,"parubnd"] = ppar.k.apply(lambda x: thickness[k])

        

    ppar = par.loc[par.parnme.str.contains("theta"),:]
    par.loc[ppar.parnme,"parlbnd"] = ppar.parval1.values - 0.3
    par.loc[ppar.parnme,"parubnd"] = ppar.parval1.values + 0.3

    ppar = par.loc[par.parnme.str.contains("kv"),:]
    par.loc[ppar.parnme,"parlbnd"] = 1e-9
    par.loc[ppar.parnme,"parubnd"] = 1e-3


    ppar = par.loc[par.parnme.str.contains("pcs0"),:]
    par.loc[ppar.parnme,"partrans"] = "none"
    par.loc[ppar.parnme,"parlbnd"] = ppar.parval1.values - 100
    par.loc[ppar.parnme,"parubnd"] = ppar.parval1.values + 100

    ppar = par.loc[par.parnme.str.contains("h0"),:]
    par.loc[ppar.parnme,"partrans"] = "none"
    par.loc[ppar.parnme,"parlbnd"] = ppar.parval1.values - 100
    par.loc[ppar.parnme,"parubnd"] = ppar.parval1.values + 100

    #set the % discrep obs to zero weight bc just a lil noise in the solution causes a big discrep
    obs = pst.observation_data
    obs.loc[obs.usecol=="percent-discrepancy","weight"] = 0.0

    #set the weight for first stress period csub obs to zero since it gets a null value
    dobs = obs.loc[obs.usecol.apply(lambda x: "precon" in x or "delay" in x and "minus" not in x),:].copy()

    dobs["datetime"] = pd.to_datetime(dobs["datetime"])
    min_dt = dobs["datetime"].min()
    zdobs = dobs.loc[dobs.datetime==min_dt,"obsnme"]
    obs.loc[zdobs,"weight"] = 0.0

    # run once to make sure phi is near zero
    pst.control_data.noptmax = 0    
    pst.write(os.path.join(template_ws, 'pest.pst'), version=2)

    pyemu.os_utils.run("pestpp-ies pest.pst",cwd=template_ws)
    pst = pyemu.Pst(os.path.join(os.path.join(template_ws,"pest.pst")))
    print(pst.phi)
    phicomps = pst.phi_components
    print([(comp,phi) for comp,phi in phicomps.items() if phi > 10])
    assert pst.phi < 1
    pst = pf.pst

    #pst.observation_data = obs_df
    pst.write(os.path.join(template_ws, 'pest.pst'), version=2)
    #pyemu.os_utils.run("pestpp-ies pest.pst",cwd=template_ws)
    #print(raw_csub_df)
    #print(raw_csub_df.cdelay.str.lower().values)
    if "delay" not in pkg_df.cdelay.str.lower().values:
        print("fixing delay-form-specific parameters")
        delay_tags = ["kv","rnb","h0"]
        par = pst.parameter_data
        for tag in delay_tags:
            par.loc[par.parnme.str.contains(tag),"partrans"] = "fixed"
        pst.parameter_data = par

    # tie the predictive ghb levels to the last historic level
    # big assumption - only one predictive stress period!
    # par = pst.parameter_data
    # gpar = par.loc[par.pname=="ghbhead",:].copy()
    # assert gpar.shape[0] > 0
    # gpar["kper"] = gpar.kper.astype(int)
    # gpar["k"] = gpar.k.astype(float).astype(int)
    # tied_parnames = []
    # for u in gpar.k.unique():
    #     ugpar = gpar.loc[gpar.k==u,:].copy()
    #     ugpar.sort_values(by="kper",inplace=True)
    #     tname = ugpar.parnme.iloc[-1]
    #     t2name = ugpar.parnme.iloc[-2]
    #     par.loc[tname,"partrans"] = "tied"
    #     par.loc[tname,"partied"] = t2name
    #     tied_parnames.append(tname)
    
    # #should be only two tied pars - only for each ghb in upper and lower aquifers
    # assert par.loc[par.partrans=="tied",:].shape[0] == len(gpar.k.unique())

    par = pst.parameter_data
    par["partied"] = np.nan
    gpar = par.loc[par.pname == "ghbhead", :].copy()
    if gpar.shape[0] > 0:
        gpar["datetime"] = pd.to_datetime(gpar.datetime, format="%Y-%m-%d")
        gpar = gpar.loc[gpar.datetime.dt.year > 2024, :]
        assert gpar.shape[0] > 0
        par.loc[gpar.parnme, "partrans"] = "fixed"

    dpar = par.loc[par.pname == "directghbhead", :].copy()
    if dpar.shape[0] > 0:
        par.loc[dpar.parnme, "partrans"] = "fixed"

    jpar = par.loc[par.pname=="monthlyghbhead",:].copy()
    if jpar.shape[0] > 0:

        jpar["datetime"] = pd.to_datetime(jpar.datetime, format="%Y-%m-%d")
        jpar.sort_values(by="datetime",inplace=True)
        jpar["month"] = jpar.datetime.dt.month
        uks = jpar.k.unique()
        uks.sort()
        for k in uks:
            ujpar = jpar.loc[jpar.k==k,:]
            umons = ujpar.month.unique()
            umons.sort()
            for umon in umons:
                upar = ujpar.loc[ujpar.month==umon,:]
                if upar.shape[0] > 1:
                    par.loc[upar.parnme.iloc[1:],'partrans'] = "tied"
                    #print(umon,upar.parnme.iloc[0])
                    #print("...",upar.parnme)
                    par.loc[upar.parnme.iloc[1:],'partied'] = upar.parnme.iloc[0]


    apar = par.loc[par.pname=="annualghbhead",:].copy()
    if apar.shape[0] > 0:

        apar["datetime"] = pd.to_datetime(apar.datetime, format="%Y-%m-%d")
        apar["year"] = apar.datetime.dt.year
        tofix = apar.loc[apar.year>2024,"parnme"]
        par.loc[tofix,"partrans"] = "fixed"
        apar.loc[tofix,"partrans"] = "fixed"
        apar = apar.loc[apar.partrans!="fixed",:]
        uks = apar.k.unique()
        uks.sort()
        for k in uks:
            uapar = apar.loc[apar.k==k,:]
            uyears = uapar.year.unique()
            uyears.sort()
            for uyear in uyears:
                upar = uapar.loc[uapar.year==uyear,:]
                if upar.shape[0] > 1:
                    par.loc[upar.parnme.iloc[1:],'partrans'] = "tied"
                    #print(uyear,upar.parnme.iloc[0])
                    par.loc[upar.parnme.iloc[1:],'partied'] = upar.parnme.iloc[0]


    pf.pst = pst
    np.random.seed(pyemu.en.SEED)
    pe = pf.draw(num_reals=num_reals, use_specsim=False) 

    gpar = par.loc[par.pname == "ghbhead", :].copy()
    gpar = gpar.loc[gpar.partrans!="fixed",:]

    if gpar.shape[0]:
        gpar["datetime"] = pd.to_datetime(gpar.datetime, format="%Y-%m-%d")
        kvals = gpar.k.unique()
        kvals.sort()
        ghb_tpar_dfs = []
        for k in kvals:
            kpar = gpar.loc[gpar.k==k,:].copy()
            kpar['x'] = (kpar.datetime - kpar.datetime.min()).dt.days
            kpar["y"] = 0.0
            ghb_tpar_dfs.append(kpar.copy())
        ghb_atpar_dfs = []
        # if apar.shape[0] > 0:
        #     kvals = apar.k.unique()
        #     kvals.sort()
        #     for k in kvals:
        #         kpar = apar.loc[apar.k==k,:].copy()
        #         kpar['x'] = (apar.datetime - apar.datetime.min()).dt.days
        #         kpar["y"] = 0.0
        #         ghb_atpar_dfs.append(kpar.copy())

        pe = pyemu.helpers.geostatistical_draws(pst=pst,struct_dict={temporal_gs:ghb_tpar_dfs},num_reals=num_reals)

    pe.enforce() 
    # this name is harded in xfer
    pe.to_binary(os.path.join(template_ws, "prior_pe.jcb")) 
    pst.pestpp_options["ies_par_en"] = "prior_pe.jcb"

    pst.write(os.path.join(template_ws, 'pest.pst'), version=2)
   
    files = ["ies_functions.py","model_functions.py","workflow.py"]
    files = [f for f in os.listdir(".") if f.endswith(".py")]
    for f in files:
        shutil.copy2(f,os.path.join(template_ws,f+".txt")) 
    shutil.copy2(os.path.join(site_name,"prep_data.py"),os.path.join(template_ws,"prep_data.py.txt"))
    shutil.copytree(os.path.join(site_name,"source_data"),os.path.join(template_ws,"source_data"))
    shutil.copytree(os.path.join(site_name,"processed_data"),os.path.join(template_ws,"processed_data"))
      


def test_enforce_ghb_botm(d):
    b_d = os.getcwd()
    os.chdir(d)
    enforce_ghb_botm()
    os.chdir(b_d)


def enforce_ghb_botm(nlay=3):
    import flopy
    #sim = flopy.mf6.MFSimulation.load()
    #gwf = sim.get_model()
    #tol = 0.1 #harded coded also in build model
    #botm_dict = {k:gwf.dis.botm.array[k,0,0]+tol for k in range(gwf.dis.nlay.data)}
    botm_dict = {k: np.loadtxt("model.dis_botm_layer{0}.txt".format(k + 1)) for k in range(nlay)}
    ghb_files = [f for f in os.listdir('.') if f.startswith('model.ghb_stress_period_data') and f.endswith(".txt")]

    for f in ghb_files:
        df = pd.read_csv(f, header=None, names=["l", "r", "c", "bhead", "cond"], sep="\\s+")
        #first try to move the ghb down a layer
        #df["l"] = [lay+1 if bh > botm_dict[lay - 1] and lay < df.l.max() else lay for lay, bh,cond in zip(df.l, df.bhead, df.cond)]
        #ok if still too low, then turn down cond and reset head
        df["cond"] = [cond if bh > botm_dict[lay - 1] else 1e-10 for lay, bh,cond in zip(df.l, df.bhead, df.cond)]
        df["bhead"] = [bh if bh > botm_dict[lay - 1] else botm_dict[lay - 1] for lay, bh in zip(df.l, df.bhead)]
        df.to_csv(f, header=False, index=False, sep=" ")
    ghb_files = [f for f in os.listdir('.') if f.startswith('model.ghb_stress_period_data') and f.endswith(".txt")]
    
    for f in ghb_files:
        df = pd.read_csv(f,header=None,names=["l","r","c","bhead","cond"],sep="\\s+")
        df["bhead"] = [bh if bh > botm_dict[lay-1] else botm_dict[lay-1] for lay,bh in zip(df.l,df.bhead) ]
        df.to_csv(f,header=False,index=False,sep=" ")

def test_ghbhead_to_csv(t_d):
    b_d = os.getcwd()
    os.chdir(t_d)
    df = ghbhead_to_csv()
    os.chdir(b_d)
    return df
    
def ghbhead_to_csv():
    ghb_files = [f for f in os.listdir('.') if f.startswith("model.ghb_stress") and f.endswith(".txt")]
    ghb_kper = [int(f.split(".")[1].split('_')[-1])-1 for f in ghb_files]
    
    import flopy
    sim = flopy.mf6.MFSimulation.load(load_only=["tdis"])
    # exit()
    # gwf = sim.get_model()
    # spd = gwf.ghb.stress_period_data
    start_datetime = sim.tdis.start_date_time.data
    
    dts = pd.to_datetime(start_datetime) + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit='d')
    dfs = []
    ddts = []
    #for data,dt in zip(spd.array,dts):
    for kper,fname in zip(ghb_kper,ghb_files):
        #df = pd.DataFrame.from_records(data)
        df = pd.read_csv(fname,header=None,names=["l","r","c","bhead","cond"],sep="\\s+")
        df["datetime"] = dts[kper]
        df["kper"] = kper
        df["fname"] = fname
        #print(dt,data)
        dfs.append(df.loc[:,["datetime","kper","l","bhead","cond","fname"]])
    df = pd.concat(dfs)
    #df["k"] = df.pop("cellid").apply(lambda x: x[0])
    df['k'] = df.pop("l") - 1
    df.sort_values(by=["k","datetime"],inplace=True)
    df.index = np.arange(df.shape[0])
    df.index.name="count"
    df.to_csv("ghb.csv")
    return df
    #print(df)

def set_obsvals_and_weights(obs_file,t_d,num_reals=100,use_focus_weights=False,
                            use_obs_diff=False,verification_window=None,
                            assimilate_all=False,prefer_less_rebound=1e-20,
                            prefer_less_delayedfuture=False):
    """helper function to assign standard deviations to pest observations.
    These are used to inform generation of obs+noise ensembles

    Args:
        mod_name (str): pest control file prefix
        m_d (str): relative path to pest master directory (see run_local())
        use_focus_weights (bool): flag to focus weighting on late time obs
        use_diff_obs (bool): flag to non-zero weight temporal diff sub obs in late
           time
        verification_window (list(datetimes)): two datetimes that mark the beginning 
           and end of a time window where subsidence observations will be zero weighted
           so that these obs can be used to verify the model's performance for unseen 
           subsidence
        assimilate_all (bool): flag to assimilate all observed subsidence values regardless
           of the value of the `should_assimilate` flag in the observed subsidence dataset
           (`should_assimilate` is set in some of the site specific prep_data() functions
           for dodgy or questionable observations)
    """

    # grp_stdev_abs_dict = {'total_subsidence': 0.5} 

    assert os.path.exists(obs_file),obs_file

    verf_start,verf_end = None,None
    if verification_window is not None:
        verf_start = pd.to_datetime(verification_window[0])
        verf_end = pd.to_datetime(verification_window[1])


    obs_file_dest = os.path.join(t_d,os.path.split(obs_file)[1])
    if os.path.exists(obs_file_dest):
        os.remove(obs_file_dest)
    #check for a sub data csv in processed data
    dtfmt = "%m/%d/%Y"
    pobs_file = obs_file.replace("source","processed")
    if os.path.exists(pobs_file):
        obs_file = pobs_file
        dtfmt = "%Y-%m-%d"

    shutil.copy2(obs_file,obs_file_dest)

    odf = pd.read_csv(obs_file_dest)
    odf.columns = [c.lower() for c in odf.columns]
    odf["datetime"] = pd.to_datetime(odf.pop("date"),format=dtfmt)
    odf["source"] = odf.source.str.lower()
    odf.sort_values(by="datetime",inplace=True)
    pst = pyemu.Pst(os.path.join(t_d, 'pest.pst'))

    obs = pst.observation_data    
    obs.loc[:,"weight"] = 0.0
    obs.loc[:,"standard_deviation"] = np.nan
    obs.loc[:,"lower_bound"] = 0.0
    obs.loc[:,"observed"] = False
    obs.loc[:,"source"] = "mf6"
    obs.loc[:,"lower_bound"] = np.nan
    obs.loc[:,"is_verf"] = False

    site = obs_file.split(os.path.sep)[0]
    ghb_obs_filename = os.path.join(site,"processed_data","{0}.orgts_data.csv".format(site))
    ghb_count = 0
    if os.path.exists(ghb_obs_filename):
        gdf = pd.read_csv(ghb_obs_filename,index_col=0,parse_dates=True)
        gdf["k"] = gdf.klayer.astype(int)
        gobs = obs.loc[obs.usecol=="bhead",:].copy()
        gobs["datetime"] = pd.to_datetime(gobs.datetime)
        gobs["k"] = gobs.k.astype(int)
        uks = gobs.k.unique()
        uks.sort()
        for k in uks:
            kdf = gdf.loc[gdf.k==k,:].copy()
            kdf.sort_index(inplace=True)
            kobs = gobs.loc[gobs.k==k,:].copy()
            kobs.sort_values(by="datetime",inplace=True)
            udts = kobs.datetime.values
            udts.sort()
            print(udts)
            # exit()
            datetimes,sub,source,count = [],[],[],[]
            start_datetimes,org_datetimes = [],[]
            should_assimilate = []

            for i,(start,end,oname) in enumerate(zip(udts[:-1],udts[1:],kobs.obsnme.values[:-1])):
                if i >= 1:
                    start = udts[i-1]
                udf = kdf.loc[kdf.index.map(lambda x: x>=start and x<end),:]
                if udf.shape[0] == 0:
                    continue
                obs.loc[oname,"obsval"] = udf.interpolated.mean()
                obs.loc[oname,"less_than"] = udf.interpolated.max() + 40
                obs.loc[oname,"greater_than"] = udf.interpolated.min() - 40
                obs.loc[oname,"weight"] = 1.0  
                obs.loc[oname,"standard_deviation"] = 5.0  
                
                ghb_count += 1


    sobs = obs.loc[obs.usecol=="sim-subsidence-ft",:].copy()
    assert sobs.shape[0] > 0
    sobs["datetime"] = pd.to_datetime(sobs.datetime,format="%Y%m%d")
    sobs.sort_values(by="datetime",inplace=True)
    udts = np.unique(sobs.datetime.values)
    udts.sort()
    print(udts)
    # exit()
    datetimes,sub,source,count = [],[],[],[]
    start_datetimes,org_datetimes = [],[]
    should_assimilate = []

    for start,end in zip(udts[:-1],udts[1:]):
        udf = odf.loc[odf.datetime.apply(lambda x: x>=start and x<end),:].copy()

        if udf.shape[0] == 0:
            continue
        print(udf)
        print(start,end)
        datetimes.append(end)
        sub.append(udf["subsidence_ft"].max())
        source.append("-".join(udf.source.unique().tolist()))
        count.append(udf.shape[0])
        start_datetimes.append(start)
        if "should_assimilate" in udf.columns:
            if True in udf.should_assimilate.values or assimilate_all:
                should_assimilate.append(True)
            else:
                should_assimilate.append(False)
        else:
            should_assimilate.append(True)
        org_datetimes.append("-".join([dt.strftime("%Y%m%d") for dt in udf.datetime.tolist()]))

    deltas = (sobs["datetime"].iloc[1:] - sobs["datetime"].iloc[:-1]).dt.days
    odf = pd.DataFrame(data={"datetime":datetimes,"subsidence_ft":sub,"source":source, "count":count,
        "start_datetime":start_datetimes,"org_datetimes":org_datetimes,"should_assimilate":should_assimilate},index=datetimes)
    odf["org_subsidence_ft"] = odf.subsidence_ft.copy()
    odf["subsidence_ft"] -= odf.subsidence_ft.min() #reset to 0.0 sub at the start
    odf.to_csv(os.path.join(t_d,"sampled_obs.csv"))
    deltas = np.array(deltas)
    
    xquants = np.array([0.0,0.33,0.66,1.0])
    yquants = np.array([np.nanquantile(odf.subsidence_ft.values,x) for x in xquants])
    nquants = ["lower_third","lower-third","middle-third","upper-third"]
    count = 0
    verf_count = 0
    skipped = 0
    too_far = []
    sub_max = -1e30
    for dt,subval,src,should_assimilate in zip(odf.datetime,odf.subsidence_ft,odf.source,odf.should_assimilate):

        sobs["distance"] = (dt - sobs.datetime).dt.days.apply(np.abs)
        if sobs.distance.min() >  (deltas.mean() * 1.5):
            print('...sub datum too far to interpolate',dt,subval,src)
            too_far.append([dt,subval,src])
            continue
        sobs.sort_values(by="distance",inplace=True)
        #print(dt,sobs.distance.describe())
        
        idxmin = sobs.obsnme.iloc[0]
        mobs = sobs.loc[[idxmin],:]
        assert mobs.shape[0] == 1
        oname = mobs.obsnme.values[0]
        sub_max = max(subval,sub_max)
        if obs.loc[oname,"observed"] == True:
            print(oname)
            print("duplicate obs for datetime: {0}".format(dt))
            continue
        obs.loc[oname,"observed"] = True
        obs.loc[oname,"obsval"] = subval
        obs.loc[oname,"source"] = src
        if should_assimilate:
            obs.loc[oname,"weight"] = 1.0
        else:
            #obs.loc[oname,"weight"] = 0.25
            skipped += 1
        obs.loc[oname,"lower_bound"] = 0.0
        for i,(x,y,n) in enumerate(zip(xquants[:-1],yquants[:-1],nquants[:-1])):
            if subval >= y and subval < yquants[i+1]:
                break

        decade = int(dt.year/10) * 10

        obs.loc[oname,"obgnme"] += "_weightquant:"+nquants[i+1]+"_decade:{0}".format(decade)
        if use_focus_weights is not None and use_focus_weights and dt.year > 2000:
            obs.loc[oname,"weight"] *= 3
        if verf_start is not None:
            if dt > verf_start and dt <= verf_end:
                obs.loc[oname,"weight"] = 0.0
                obs.loc[oname,"is_verf"] = True
                verf_count += 1
                if not should_assimilate:
                    skipped -= 1
        if dt.year < 1950 and odf["subsidence_ft"].max() > 5.0:
            obs.loc[oname,"standard_deviation"] = 0.15
        elif "ext" not in src:
            obs.loc[oname,"standard_deviation"] = 0.05#max(0.0001,subval * 0.25)
            # obs.loc[oname, "standard_deviation"] = 0.05 #testLessNoise
        elif "ext" in src:
            obs.loc[oname,"standard_deviation"] = 0.05#max(0.0001,subval * 0.25)
            # obs.loc[oname, "standard_deviation"] = 0.05 #testLessNoise
        else:
            raise Exception("unknown source: {0}".format(src))
        count += 1
    
    print(count," obs set")

    if len(too_far) > 0:
        print("WARNING: the following sub obs were too far in time for usage:")
        for tf in too_far:
            print(",".join([str(t) for t in tf]))
    
    assert count > 0
    print("pst.nnz_obs",pst.nnz_obs)
    print("verf_count",verf_count)
    print("skipper",skipped)
    print("count",count)
    assert pst.nnz_obs + verf_count + skipped == count + ghb_count
    if pst.nnz_obs - ghb_count > odf.shape[0]:
        raise Exception("something is wrong")
    if pst.nnz_obs != odf.shape[0]:
        print("WARNING!!! not all observations mapped to a pest control file observation...")
    

    # now set the difference obsvals
    nzobs = obs.loc[obs.observed==True,:].copy()
    nzobs["datetime"] = pd.to_datetime(nzobs.datetime)
    nzobs.sort_values(by="datetime",inplace=True)
    dobs = obs.loc[obs.oname=="subdiff",:].copy()
    dobs["dt1"] = pd.to_datetime(dobs.dt1)
    dobs["dt2"] = pd.to_datetime(dobs.dt2)
    dcount = 0
    for i in range(nzobs.shape[0]-1):
        o1 = nzobs.obsnme.iloc[i]
        v1 = nzobs.obsval.iloc[i]
        dt1 = nzobs.datetime.iloc[i]
        if dt1.year < 2010:
            continue
        if verf_start is not None:
            if dt1 > verf_start and dt1 <= verf_end:
                continue

            
        for ii in range(i+1,nzobs.shape[0]):
            o2 = nzobs.obsnme.iloc[ii]
            v2 = nzobs.obsval.iloc[ii]
            dt2 = nzobs.datetime.iloc[ii]
            if verf_start is not None:
                if dt2 > verf_start and dt2 <= verf_end:
                    continue
            dname = dobs.loc[(dobs.dt1==dt1) & (dobs.dt2==dt2),"obsnme"]
            assert dname.shape[0] == 1
            obs.loc[dname,"obsval"] = v2 - v1
            obs.loc[dname,"observed"] = True
            obs.loc[dname,"standard_deviation"] = max(0.00001,(v2 - v1)*0.01)
            # obs.loc[dname, "standard_deviation"] = 0.000001 #testLessNoise
            if use_obs_diff:
                obs.loc[dname,"weight"] = 1.0
                if use_focus_weights and dt2.year > 2010:
                    obs.loc[dname,"weight"] = 100
                #if dt1.year > 2010:
                #    obs.loc[dname,"weight"] *= 3
                
                
                dcount += 1
    print(dcount," difference obs set")
    #if prefer_less_rebound:    
    cobs = obs.loc[obs.usecol.str.startswith("tdif-compaction."),:].copy()
    assert cobs.shape[0] > 0
    obs.loc[cobs.index,"obsval"] = 0.0
    obs.loc[cobs.index,"obgnme"] = "greater_than_rebound"
    obs.loc[cobs.index,"weight"] = 1

    # cobs = obs.loc[obs.usecol.str.startswith("compaction"),:].copy()
    # cobs["datetime"] = pd.to_datetime(cobs.datetime)
    # print(cobs.usecol.unique())
    # cobs["layer"] = cobs.usecol.apply(lambda x: int(x.split(".")[1]))
    # total = cobs.groupby("datetime").sum()
    # cobs = cobs.loc[cobs.apply(lambda x: x.datetime.year<=2024 and x.layer == 1,axis=1),:]
    # assert cobs.shape[0] > 0
    # cobs.sort_values(by="datetime",inplace=True)
    # obs.loc[cobs.index,"obsval"] = 0.5
    # obs.loc[cobs.index,"obgnme"] = "less_than"
    # obs.loc[cobs.index,"weight"] = 100.0
    
    iobs = obs.loc[obs.usecol=="implied-thick-diff",:]
    assert iobs.shape[0] > 0
    obs.loc[iobs.obsnme,"obsval"] = 0.0
    obs.loc[iobs.obsnme,"obgnme"] = "claydiff"
    obs.loc[iobs.obsnme,"weight"] = 1.0
    obs.loc[iobs.obsnme,"standard_deviation"] = 0.001

    if prefer_less_delayedfuture:
        ht_dt1_year = 2024#pd.to_datetime("12-31-2024")
        ht_dt2_year = 2053#pd.to_datetime("12-31-2053")
        sobs = obs.loc[obs.oname=="subdiff",:].copy() 
        sobs["dt1"] = pd.to_datetime(sobs.dt1)
        sobs["dt2"] = pd.to_datetime(sobs.dt2)

        sobs = sobs.loc[sobs.dt1.dt.year==ht_dt1_year,:]
        assert sobs.shape[0] > 0
        sobs = sobs.loc[sobs.dt2.dt.year==ht_dt2_year,:]
        assert sobs.shape[0] > 0

        sobs.sort_values(by=["dt1","dt2"],ascending=False,inplace=True)
        ht_oname = sobs.obsnme.iloc[0]
          
        obs.loc[ht_oname,"weight"] = 1.0 
        obs.loc[ht_oname,"obgnme"] = "less_than_htobs"
        obs.loc[ht_oname,"obsval"] = sub_max * 0.1
        obs.loc[ht_oname,"standard_deviation"] = 0.001
        

    nzgrps = [g for g in pst.nnz_obs_groups if "sim-subsidence-ft" in g]
    assert len(nzgrps) > 0
    struct_dict = {}
    obs["distance"] = np.nan
    obs["org_group"] = obs.obgnme.values
    obs.loc[obs.obgnme.isin(nzgrps),"obgnme"] = "sim-subsidence-ft"
    nzgrps = ["sim-subsidence-ft"]
    for grp in nzgrps:
        tobs = obs.loc[obs.obgnme==grp,:].copy()
        tobs = tobs.loc[tobs.weight >0,:]
        assert tobs.shape[0] > 0
        v = pyemu.geostats.ExpVario(contribution=1.0,a=7300)
        gs = pyemu.geostats.GeoStruct(variograms=[v],name=grp)
        tag_names = tobs.obsnme.tolist()
        tobs["datetime"] = pd.to_datetime(tobs.datetime,format="%Y%m%d")
        start = tobs.datetime.min()
        obs.loc[tag_names,"distance"] = (tobs.datetime - start).dt.days
        struct_dict[gs] = tobs.obsnme.tolist()
        #print(grp)


    #print(pst.observation_data.loc[pst.nnz_obs_names,"distance"])
    oe = pyemu.helpers.autocorrelated_draw(pst,struct_dict,enforce_bounds=True,num_reals=num_reals)
    oe = oe.loc[:,pst.nnz_obs_names]
    oe.to_binary(os.path.join(t_d,"noise.jcb"))
    pst.pestpp_options["ies_obs_en"] = "noise.jcb"
    obs["obgnme"] = obs.org_group
    pst.observation_data = obs

    if use_focus_weights is not None and not use_focus_weights:
        with open(os.path.join(t_d,"phi.csv"),'w') as f:

            f.write("subdiff,0.1\n")
            f.write("lower-third,{0}\n".format(0.3))
            f.write("middle-third,{0}\n".format(0.3))
            f.write("upper-third,{0}\n".format(0.3))
            f.write("greater_than_rebound,{0}\n".format(prefer_less_rebound))
            f.write("claydiff,{0}\n".format(0.15))
            f.write("less_than_htobs,{0}\n".format(0.15))
            f.write("ghbhead,0.1\n")
            # f.write("subdiff,1e-20\n")
            # f.write("lower-third,1e-20\n")
            # f.write("middle-third,1e-20\n")
            # f.write("upper-third,1e-20\n")
            # f.write("greater_than_rebound,1e-20\n")
            # f.write("claydiff,1e-20\n")
            # f.write("less_than_htobs,1e-20\n")
            # f.write("ghbhead,1.0\n")
                    
        pst.pestpp_options["ies_phi_factor_file"] = "phi.csv"

    pst.control_data.noptmax = -2
    # save binary formats
    pst.pestpp_options["ies_save_binary"] = True
    pst.write(os.path.join(t_d, "pest.pst"),version=2)

    pyemu.os_utils.run("pestpp-ies pest.pst",cwd=os.path.join(t_d))
    pst = pyemu.Pst(os.path.join(t_d,"pest.pst"))
    assert pst.phi > 1e-6 #greater than numerical noise
    

def get_lst_budget(ws='.',start_datetime=None, casename='model'):
    """get the inc and cum dataframes from a MODFLOW-6 list file
    Parameters
    ----------
    ws : str
        path to the model workspace
    start_datetime : str
        a string that can be parsed by pandas.to_datetime
    Returns
    -------
    inc : pandas.DataFrame
        the incremental budget
    cum : pandas.DataFrame
        the cumulative budget
        """
    print("postprocessing lst obs...")
    import flopy
    lst_file = os.path.join(ws,f"{casename}.lst")
    assert os.path.exists(lst_file), f"{lst_file} not found"
    nfail = 0
    with open(lst_file,'r') as f:
        for line in f:
            if "FAILED TO MEET SOLVER CONVERGENCE CRITERIA" in line:
                nfail += 1
    with open(os.path.join(ws,"nfail.csv"),'w') as f:
        f.write('onme,oval\n')
        f.write('nfail,{0}\n'.format(nfail))
    lst = flopy.utils.Mf6ListBudget(os.path.join(ws,f"{casename}.lst"))
    inc,cum = lst.get_dataframes(diff=True,start_datetime=start_datetime)
    inc.columns = inc.columns.map(lambda x: x.lower().replace("_","-"))
    cum.columns = cum.columns.map(lambda x: x.lower().replace("_", "-"))
    if start_datetime is not None:
        inc.index = inc.index.strftime("%Y%m%d")
        cum.index = cum.index.strftime("%Y%m%d")
        inc.index.name = "datetime"
        cum.index.name = "datetime"
    else:
        inc.index.name = "time"
        cum.index.name = "time"
    inc.to_csv(os.path.join(ws,"inc.csv"))
    cum.to_csv(os.path.join(ws,"cum.csv"))
    return inc, cum



import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
from matplotlib.pyplot import cm
years10 = mdates.YearLocator(10)
years20 = mdates.YearLocator(20)
years1 = mdates.YearLocator()
years_fmt = mdates.DateFormatter('%Y')



def _set_axis_style(ax, labels):
    """helper function to assign style parameters to matplotlib axis object
    Args:
        ax (obj): matplotlib axis object
        labels (list): list of strings to use as axis tick labels
    """
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.5)
    
    
    
def plot_phi(m_d):

    phis = pd.read_csv(os.path.join(m_d, 'pest.phi.actual.csv'),index_col=0)
    
    # pst = pyemu.Pst(os.path.join(m_d, "pest.pst"))
    
    with PdfPages(os.path.join(m_d, "phi.pdf")) as pdf:  

        fig,axes = plt.subplots(2,1,figsize=(6,6))
        ax = axes[0]
        
        [ax.plot(phis.index.values,np.log10(phis.iloc[:,i].values),"0.5") for i in range(6,phis.shape[1])]
        ax.set_title("phi vs iteration",loc="left")
    
        ax = axes[1]
        colors = ["0.5","b"]
        # icolor = 0
        
        for i,itr in enumerate([0,phis.shape[0]-1]):
            print(itr,i)
            ax.hist(np.log10(phis.iloc[itr,6:].values),bins=20,alpha=0.5,facecolor=colors[i])
        ax.set_title("prior and posterior phi histograms",loc="left")
    
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)


            
    
def plot_en_compaction(m_d, noptmax=None):
    
    
    # test = pd.read_csv(os.path.join(site_name, 'model_ws', 'baseline', 'J88.baseline.csub.obs.csv'))
    # add_cols = [c for c in test.columns if 'COMPACTION' in c]
    # test.loc[:, 'sim_subsidence_ft'] = test.loc[:, add_cols].sum(axis=1)

    pst = pyemu.Pst(os.path.join(m_d, "pest.pst"))
    if noptmax is None:
        phidf = pd.read_csv(os.path.join(m_d, "pest.phi.actual.csv"))
        noptmax = phidf.iteration.max()
        # noptmax = "combined"
    pr_en = pyemu.ObservationEnsemble.from_binary(pst,
                                                  os.path.join(m_d,
                                                               'pest.0.obs.jcb'))

    
    pt_en = pyemu.ObservationEnsemble.from_binary(pst,
                                                  os.path.join(m_d,
                                                               f'pest.{noptmax}.obs.jcb'))
    
    assert pt_en.shape[1] == pst.nobs
    noise_en = pyemu.ObservationEnsemble.from_binary(pst,
                                                    os.path.join(m_d,
                                                                'pest.obs+noise.jcb'))
 
    obs = pst.observation_data
    sobs = obs.loc[obs.usecol=="sim-subsidence-ft",:].copy()
    assert sobs.shape[0] > 0
    sobs["datetime"] = pd.to_datetime(sobs.datetime)
    cobs = obs.loc[obs.usecol.str.startswith("compaction."),:].copy()
    assert cobs.shape[0] > 0
    tdcobs = obs.loc[obs.usecol.str.startswith("tdif-compaction."),:].copy()
    assert cobs.shape[0] > 0
    sobs["datetime"] = pd.to_datetime(sobs.datetime,format="%Y%m%d")
    cobs["datetime"] = pd.to_datetime(cobs.datetime,format="%Y%m%d")
    tdcobs["datetime"] = pd.to_datetime(tdcobs.datetime)
    
    
    pobs = obs.loc[obs.usecol.str.contains("preconstress"),:].copy()
    assert pobs.shape[0] > 0

    pobs["datetime"] = pd.to_datetime(pobs.datetime,format="%Y%m%d")

    hobs = obs.loc[obs.oname=="hds",:].copy()
    assert hobs.shape[0] > 0
    hobs["k"] = hobs.usecol.apply(lambda x: int(x.split(".")[1])) - 1
    hobs["datetime"] = pd.to_datetime(hobs.datetime)

    dobs = obs.loc[obs.usecol.str.startswith("delay-head"),:]
    assert dobs.shape[0] > 0
    dobs["k"] = dobs.usecol.apply(lambda x: int(x.split(".")[1])) - 1
    dobs["delay_node"] = dobs.usecol.apply(lambda x: int(x.split(".")[2])) - 1
    
    dobs["datetime"] = pd.to_datetime(dobs.datetime)

    obsobs = sobs.loc[sobs.observed==True,:].copy()
    nzobs = sobs.loc[sobs.weight > 0, :].copy()

    nzobs.sort_values(by="datetime",inplace=True)
    obsobs.sort_values(by="datetime", inplace=True)

    #pdc = pd.read_csv(os.path.join(m_d, f'{pst_name.split(".pst")[0]}.pdc.csv'))
    
    gobs = obs.loc[obs.oname=="ghbhead",:].copy()
    gobs["datetime"] = pd.to_datetime(gobs.datetime)
    gobs["k"] = gobs.k.astype(int)
    uks = gobs.k.unique()
    uks.sort()
    site_name= os.path.split(m_d)[0]
    
    raw_gw_df = pd.read_csv(os.path.join(site_name,"processed_data","{0}.ts_data.csv".format(site_name)),index_col=0,parse_dates=True)
    raw_gw_df = raw_gw_df.loc[pd.notna(raw_gw_df.Alt),:]

    with PdfPages(os.path.join(m_d, f"compaction_summary_{noptmax}.pdf")) as pdf:  
        fig, axes = plt.subplots(len(uks)+2, 1, figsize=(15,15))
            
        print(pt_en.columns)
        print(sobs.obsnme)
        ymax = max(obsobs.obsval.max(),pt_en.loc[:,sobs.obsnme].values.max())#np.round((nzobs.obsval.max() + 10.)/10.) * 10.
        ymin = min(obsobs.obsval.min(), 0)
        for uk,ax in zip(uks,axes[2:2+len(uks)]):
            uobs = gobs.loc[gobs.k==uk,:].copy()
            uobs.sort_values(by="datetime",inplace=True)
            dts = uobs.datetime
            vals = pr_en.loc[:,uobs.obsnme].values
            [ax.plot(dts,vals[i,:],"0.5",alpha=0.05,lw=0.05,label="prior reals") for i in range(vals.shape[0])]
            vals = pt_en.loc[:,uobs.obsnme].values
            [ax.plot(dts,vals[i,:],"b",alpha=0.05,lw=0.05,label="posterior reals") for i in range(vals.shape[0])]
            if "base" in pr_en.index:
                ax.plot(dts,pr_en.loc["base",uobs.obsnme].values,"k",alpha=0.5,lw=4.0,zorder=10,label="prior base real")
            #if "base" in pt_en.index:
            #    ax.plot(dts,pt_en.loc["base",uobs.obsnme].values,"m--",alpha=0.5,lw=1.0,zorder=10)
            
            kraw = raw_gw_df.loc[raw_gw_df.klayer==uk,:].copy()
            kraw.sort_index(inplace=True)
            ax.scatter(kraw.index.values,kraw.Alt.values,marker='o',s=10,c='k',label="org gw level")
            ax.set_title("ghb head layer {0}".format(uk+1),loc="left")
    

        # ax = axes[2+len(uks)]
        # ax.set_title("change in critical stress",loc="left")
        # usecols = pobs.usecol.unique()
        # usecols.sort()
        # color = cm.jet(np.linspace(0, 1, len(usecols))) 
        # for i,usecol in enumerate(usecols):
        #     if len(usecol.split(".")) > 1:
        #         continue
        #     ucobs = pobs.loc[pobs.usecol==usecol,:].copy()
        #     ucobs.sort_values(by="datetime",inplace=True)
        #     dts = ucobs.datetime.values
        #     vals = pt_en.loc[:,ucobs.obsnme].values
        #     vals[vals==-999] = np.nan
        #     vals = vals[:,0:-1] - vals[:,1:]

        #     #vals[np.abs(vals)>1e10] = np.nan
            
        #     print(vals.shape,dts.shape)
        #     vals = np.abs(vals)
        #     [ax.plot(dts[:-1], vals[ii,:], color=color[i], lw=0.125, label=usecol, alpha=0.5) for ii in range(vals.shape[0])]
        #     print(ax.get_ylim())   
        
                     
        # plot non-zero weighted observations by measurement type
        color = cm.jet(np.linspace(0, 1, len(obsobs.source.unique())))

        ax = axes[0]
        dts = nzobs.datetime.values
        vals = noise_en.loc[:,nzobs.obsnme].values
        [ax.plot(dts, vals[i,:], color='r', lw=0.025, alpha=0.1,label="noise reals") for i in range(vals.shape[0])]
        

        dts = sobs.datetime.values
        vals = pr_en.loc[:,sobs.obsnme].values
        [ax.plot(dts, vals[i,:], color='0.5', lw=0.025, alpha=0.1,
                 label="prior reals") for i in range(vals.shape[0])]
        vals = pt_en.loc[:,sobs.obsnme].values
        [ax.plot(dts, vals[i, :], color='b', lw=0.025, alpha=0.1,
                 label='posterior reals') for i in range(vals.shape[0])]

        
        # if the "base" realization is found, plot it with a heavier line
       
        if "base" in pt_en.index:
            axes[0].plot(dts, pt_en.loc["base", sobs.obsnme].values, color='b', lw=2.5,
                    label='base posterior')
            axes[1].plot(dts, pt_en.loc["base", sobs.obsnme].values, color='b', lw=2.5,
                    label='base posterior')

            base_series = pd.DataFrame(data={"subsidence_ft":pt_en.loc["base",sobs.obsnme].values},index=dts)
            base_series.index.name = "datetime"
            base_series.to_csv(os.path.join(m_d,"baseseries_{0}.csv".format(noptmax)))


        else:
            [axes[1].plot(dts, vals[i, :], color='b', lw=0.025, alpha=0.1,
                 label='posterior reals') for i in range(vals.shape[0])]

        mean_series = pd.DataFrame(data={"subsidence_ft":pt_en.loc[:,sobs.obsnme].mean().values},index=dts)
        mean_series.index.name = "datetime"
        mean_series.to_csv(os.path.join(m_d,"meanseries_{0}.csv".format(noptmax)))


        if "base" in pr_en.index:
            axes[0].plot(dts, pr_en.loc["base", sobs.obsnme], color='0.5', lw=2.5,
                                label='base prior')
            axes[1].plot(dts, pr_en.loc["base", sobs.obsnme], color='0.5', lw=2.5,
                                label='base prior')

        for i,mtype in enumerate(obsobs.source.unique()):
            tmp = nzobs.loc[nzobs.source==mtype,:].copy()
            tmp = tmp.loc[tmp.weight > 0, :]
            #if tmp.shape[0] > 0:
            tmp.sort_values(by="datetime", inplace=True)
            axes[1].scatter(tmp.datetime.values, tmp.obsval.values, marker="o", c=color[i], s=4,
                    label=mtype,zorder=10)
            axes[0].scatter(tmp.datetime.values, tmp.obsval.values, marker="o", c='r', s=4,
                    label="assimilated",zorder=10)

            tmp2 = obsobs.loc[obsobs.source == mtype, :].copy()
            tmp2.sort_values(by="datetime", inplace=True)
            diff = list(set(tmp.obsnme.tolist()).symmetric_difference(tmp2.obsnme.tolist()))

            tmpd = obsobs.loc[diff,:].copy()
            tmpd.sort_values(by="datetime",inplace=True)

            axes[1].scatter(tmpd.datetime.values, tmpd.obsval.values, marker="o", facecolors="none",alpha=0.5,edgecolors=color[i], s=50,
                            zorder=10)
            axes[0].scatter(tmpd.datetime.values, tmpd.obsval.values, marker="o", s=50, alpha=0.5,edgecolors="r",facecolors="none",
                            label="observed", zorder=10)
        
        axes[0].set_title('total compaction obs vs sim with noise', loc="left")
        axes[1].set_title('different data types with posterior reals', loc="left")
        axes[0].set_ylim(ymin,ymax)
        axes[1].set_ylim(ymin,ymax)
        xlim = axes[0].get_xlim()

        for iax,ax in enumerate(axes):
            handles, labels = ax.get_legend_handles_labels()
            handelz, labelz = [], []
            for i,l in enumerate(labels):
                if l not in labelz:
                    labelz.append(l)
                    handelz.append(handles[i])
            ax.legend(handles=handelz, labels=labelz)    
            ax.set_xlim(xlim)
            ax.grid()
            #if iax < 2:
            #    ax.set_ylim(ymin,ymax)#ylim)
            
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)
        
        uks = dobs.k.unique()
        uks.sort()
        critical_results = {}
        mean_critical_results = {}
        delay_ib_lowestgw = {}
        gwfminusibgwl_df = {}
        for uk in uks:
            uhobs = hobs.loc[hobs.k==uk,:].copy()
            uhobs.sort_values(by="datetime",inplace=True)
            uudobs = dobs.loc[dobs.k==uk,:].copy()
            unodes = uudobs.delay_node.unique()
            unodes.sort()
            fig,axes = plt.subplots(4,1,figsize=(11,8.5))
            for node in unodes:
                pt_color="b"
                if node != unodes.min() and node != unodes.max():
                    continue
                if node == unodes.min():
                    pt_color = "m"
                else:
                    continue
                #elif node == unodes.max():
                #    pt_color = "r"

                udobs = uudobs.loc[uudobs.delay_node==node,:].copy()
                udobs.sort_values(by="datetime",inplace=True) 
                ax = axes[0]
                hdts = uhobs.datetime.values
                hvals = pt_en.loc[:,uhobs.obsnme].values
                #[ax.plot(hdts,hvals[i,:],"0.5",lw=0.1,alpha=0.3) for i in range(hvals.shape)]
                ddts = udobs.datetime.values
                dvals = pr_en.loc[:,udobs.obsnme].values
                #dvals[np.abs(dvals)>1e30] = np.nan
                dvals[dvals==-999] = np.nan
                prlowest = []
                for i in range(dvals.shape[0]):
                    lowest = []
                    ddvals = dvals[i,:]
                    for j in range(1,ddvals.shape[0]):
                        lowest.append(np.nanmin(ddvals[:j]))
                    prlowest.append(np.array(lowest))
                prlowest = np.array(prlowest)
                #prlowest = np.array([dvals[:i,:].min(axis=0).flatten() for i in range(1,dvals.shape[0])])
                #prlowest[np.abs(prlowest)>1e30]= np.nan
                
                [ax.plot(ddts,dvals[i,:],"0.5",lw=0.1,alpha=0.3) for i in range(dvals.shape[0])]
                
                dvals = pt_en.loc[:,udobs.obsnme].values
                #dvals[np.abs(dvals)>1e30] = np.nan
                dvals[dvals==-999] = np.nan

                #ptlowest = np.array([dvals[:i,:].min(axis=0).flatten() for i in range(1,dvals.shape[0])])
                #ptlowest[np.abs(ptlowest)>1e30]= np.nan
                ptlowest = []
                for i in range(dvals.shape[0]):
                    lowest = []
                    ddvals = dvals[i,:]
                    for j in range(1,ddvals.shape[0]):
                        lowest.append(np.nanmin(ddvals[:j]))
                    ptlowest.append(np.array(lowest))
                ptlowest = np.array(ptlowest)

                [ax.plot(ddts,dvals[i,:],pt_color,lw=0.1,alpha=0.3) for i in range(dvals.shape[0])]
                ax.set_title("delay ib layer {0} gw level".format(uk+1),loc="left")
                #ax.grid()

                ax = axes[1]
                [ax.plot(ddts[1:],prlowest[i,:],"0.5",lw=0.1,alpha=0.3) for i in range(prlowest.shape[0])]
                [ax.plot(ddts[1:],ptlowest[i,:],pt_color,lw=0.1,alpha=0.3) for i in range(ptlowest.shape[0])]
                ax.set_title("delay ib layer {0} lowest gw level".format(uk+1),loc="left")
                #ax.grid()

                ax = axes[2]
                print(hvals.shape,dvals.shape)
                [ax.plot(ddts,hvals[i,:] - dvals[i,:],pt_color,lw=0.2,alpha=0.4) for i in range(dvals.shape[0])]
                ax.set_title("gwf minus delay ib gw level layer {0}".format(uk+1),loc="left")
                #ax.grid()

                ax = axes[3]
                ptlowest_diff = hvals[:,1:] - ptlowest
                [ax.plot(ddts[1:],ptlowest_diff[i,:],pt_color,lw=0.2,alpha=0.4) for i in range(ptlowest_diff.shape[0])]
                ax.set_title(" gwf gw level minus delay ib lowest gw level layer {0}".format(uk+1),loc="left")
                #ax.grid()
                if "base" in pt_en.index:
                    bidx = pt_en.index.tolist().index("base")
                    critical_results["layer_{0}".format(uk+1)] = pd.Series(ptlowest_diff[bidx,:],index=ddts[1:])
                    print(node,pt_color,ptlowest_diff[bidx,:])
                    ax.plot(ddts[1:],ptlowest_diff[bidx,:],pt_color,lw=3.5,alpha=1)
                    
                mean_critical_results["layer_{0}".format(uk+1)] = pd.Series(ptlowest_diff.mean(axis=0),index=ddts[1:])
                delay_ib_lowestgw["layer_{0}".format(uk + 1)] = pd.Series(ptlowest.mean(axis=0), index=ddts[1:])
                gwfminusibgwl = hvals - dvals
                gwfminusibgwl_df["layer_{0}".format(uk + 1)] = pd.Series(gwfminusibgwl.mean(axis=0), index=ddts)
           
            for ax in axes:
                ax.grid()

            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
        
        if len(critical_results) > 0:
            print(critical_results)
            df = pd.DataFrame(critical_results)
            df.index.name = "datetime"
            df.to_csv(os.path.join(m_d,"critical_results_{0}.csv".format(noptmax)))
        df = pd.DataFrame(mean_critical_results)
        df.index.name = "datetime"
        df.to_csv(os.path.join(m_d,"mean_critical_results_{0}.csv".format(noptmax)))

        df = pd.DataFrame(delay_ib_lowestgw)
        df.index.name = "datetime"
        df.to_csv(os.path.join(m_d, "mean_delayib_LowestGWL_{0}.csv".format(noptmax)))

        df = pd.DataFrame(gwfminusibgwl_df)
        df.index.name = "datetime"
        df.to_csv(os.path.join(m_d, "mean_gwf_minus_ibgwl_{0}.csv".format(noptmax)))

        usecols = cobs.usecol.unique()
        fig, axes = plt.subplots(len(usecols), 1, figsize=(11,8.5))
        usecols.sort()
        base_data, mean_data = {},{}
         
        for i,usecol in enumerate(usecols):
            ax = axes[i]
            ucobs = cobs.loc[cobs.usecol==usecol,:].copy()
            ucobs.sort_values(by="datetime",inplace=True)
            dts = ucobs.datetime.values
            vals = pt_en.loc[:,ucobs.obsnme].values
            [ax.plot(dts, vals[ii,:], color="b", lw=0.125, label=usecol, alpha=0.5) for ii in range(vals.shape[0])]
            ax.set_title(usecol)
            ax.set_xlim(xlim)
            ax.grid()
            if usecol.startswith("compaction"):
                if "base" in pt_en.index:
                    base_data[usecol] = pt_en.loc["base",ucobs.obsnme].values
                mean_data[usecol] = pt_en.loc[:,ucobs.obsnme].mean().values

        if len(base_data) > 0:
            base_df = pd.DataFrame(base_data,index=dts)
            base_df["subsidence_ft"] = base_series.subsidence_ft
            base_df.index.name = "datetime"
            base_df.to_csv(os.path.join(m_d,"basedata_{0}.csv".format(noptmax)))


        mean_df = pd.DataFrame(mean_data,index=dts)
        mean_df["subsidence_ft"] = mean_series.subsidence_ft
        mean_df.to_csv(os.path.join(m_d,"meandata_{0}.csv".format(noptmax)))
        


        mx = -1e30
        for ax in axes:
            mx = max(mx,ax.get_ylim()[1])

        for ax in axes:
            ax.set_ylim(0,mx)
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)


        usecols = tdcobs.usecol.unique()
        fig, axes = plt.subplots(len(usecols), 1, figsize=(11,8.5))
        usecols.sort()
        base_data, mean_data = {},{}
         
        for i,usecol in enumerate(usecols):
            ax = axes[i]
            ucobs = tdcobs.loc[tdcobs.usecol==usecol,:].copy()
            ucobs.sort_values(by="datetime",inplace=True)
            dts = ucobs.datetime.values
            #vals = pr_en.loc[:,ucobs.obsnme].values
            #[ax.plot(dts, vals[ii,:], color="0.5", lw=0.125, label=usecol, alpha=0.35) for ii in range(vals.shape[0])]
            
            vals = pt_en.loc[:,ucobs.obsnme].values
            [ax.plot(dts, vals[ii,:], color="b", lw=0.125, label=usecol, alpha=0.5) for ii in range(vals.shape[0])]
            ax.set_title(usecol)
            ax.set_xlim(xlim)
            ax.grid()
            
        mx = -1e30
        mn = 1e30
        for ax in axes:
            mx = max(mx,ax.get_ylim()[1])
            mn = min(mn,ax.get_ylim()[0])

        for ax in axes:
            ax.set_ylim(mn,mx)
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)


        usecols = pobs.usecol.unique()
        usecols.sort()
        fig, axes = plt.subplots(len(usecols), 1, figsize=(11,25))
        for i,usecol in enumerate(usecols):
            ax = axes[i]
            ucobs = pobs.loc[pobs.usecol==usecol,:].copy()
            ucobs.sort_values(by="datetime",inplace=True)
            dts = ucobs.datetime.values
            vals = pt_en.loc[:,ucobs.obsnme].values
            dvals = vals[:,0:-1] - vals[:,1:]

            #dvals[np.abs(dvals)>1e10] = np.nan
            #vals[np.abs(vals)>1e10] = np.nan
            dvals[dvals == -999] = np.nan
            vals[vals == -999] = np.nan
            dvals = np.abs(dvals)
            [ax.plot(dts, vals[ii,:], color='b', lw=0.125, label=usecol, alpha=0.5) for ii in range(vals.shape[0])]
            
            axt = ax.twinx()
            [axt.plot(dts[:-1], dvals[ii,:],'k-',alpha=0.5,lw=0.5) for ii in range(dvals.shape[0])]
            axt.set_ylim(np.nanmax(dvals)*3.0,0)
            axt.set_ylabel("change in critical stress")
            ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*1.5)
            ax.set_title(usecol,loc="left")
            ax.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)   


def plot_par_summary(m_d,noptmax=None):
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    if noptmax is None:
        phidf = pd.read_csv(os.path.join(m_d, "pest.phi.actual.csv"))
        noptmax = phidf.iteration.max()
        # noptmax = "combined"
    pr = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.0.par.jcb"))
    pt = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.{0}.par.jcb".format(noptmax)))

    par = pst.parameter_data
    ghbpar = par.loc[par.pname=="ghbhead",:].copy()
    if ghbpar.shape[0] > 0:
        assert ghbpar.shape[0] > 0
        ghbpar["datetime"] = pd.to_datetime(ghbpar.datetime)
        ghbpar["k"] = ghbpar.k.astype(int)
        sghb = set(ghbpar.parnme.tolist())
        otherpar = par.loc[par.parnme.apply(lambda x: x not in sghb),:]
        uks = ghbpar.k.unique()
        uks.sort()
    else:
        uks = []
        otherpar = par.copy()
    
    annghbpar = par.loc[par.pname=="annualghbhead",:].copy()
    annghbpar = annghbpar.loc[annghbpar.partrans=="none"]
    if annghbpar.shape[0] > 0:
        annghbpar["datetime"] = pd.to_datetime(annghbpar.datetime)
        annghbpar["year"] = annghbpar.datetime.dt.year
        annghbpar["k"] = annghbpar.k.astype(int)
        annuks = annghbpar.k.unique()
        aghb = set(annghbpar.parnme.tolist())
        annuks.sort()
        otherpar = otherpar.loc[otherpar.parnme.apply(lambda x: x not in aghb),:]
    else:
        annuks = []

    # monghbpar = par.loc[par.pname=="monthlyghbhead",:].copy()
    # monghbpar = monghbpar.loc[monghbpar.partrans=="none"]
    # if monghbpar.shape[0] > 0:
    #     monghbpar["datetime"] = pd.to_datetime(monghbpar.datetime)
    #     monghbpar["month"] = monghbpar.datetime.month
    #     monghbpar["k"] = monghbpar.k.astype(int)
    #     monuks = monghbpar.k.unique()
    #     umons = monghbpar.month.unique()
    #     umons.sort()
    #     mghb = set(monghbpar.parnme.tolist())
    #     monuks.sort()
    #     otherpar ƒ= otherpar.loc[otherpar.parnme.apply(lambda x: x not in mghb),:]
    # else:
    #     monuks = []

    otherpar = otherpar.loc[otherpar.partrans.apply(lambda x: x in ["none","log"]),:]
    otherpar.sort_index(inplace=True)
    #print(otherpar.parnme.tolist())

    with PdfPages(os.path.join(m_d,"par_summary_{0}.pdf".format(noptmax))) as pdf:
        if len(uks) > 0:
            fig,axes = plt.subplots(len(uks),1,figsize=(10,10))
            if not isinstance(axes,np.ndarray):
                axes = [axes]
            for i,uk in enumerate(uks):
                upar = ghbpar.loc[ghbpar.k==uk,:].copy()
                upar.sort_values(by="datetime",inplace=True)
                ax = axes[i]
                dts = upar.datetime.values
                vals = pr.loc[:,upar.parnme].values
                [ax.plot(dts,vals[i,:],"0.5",lw=0.1,alpha=0.5) for i in range(vals.shape[0])]
                vals = pt.loc[:,upar.parnme].values
                [ax.plot(dts,vals[i,:],"b",lw=0.1,alpha=0.5) for i in range(vals.shape[0])]
                ax.set_ylabel("ghb head")
                ax.set_title("ghb head layer {0}".format(uk+1),loc="left")
                ax.grid()
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

        if len(annuks) > 0:
            fig,axes = plt.subplots(len(annuks),1,figsize=(10,10))
            if len(annuks) == 1:
                axes = [axes]
            for i,uk in enumerate(annuks):
                upar = annghbpar.loc[annghbpar.k==uk,:].copy()
                upar.sort_values(by="datetime",inplace=True)
                ax = axes[i]
                dts = upar.datetime.values
                vals = pr.loc[:,upar.parnme].values
                [ax.plot(dts,vals[i,:],"0.5",lw=0.1,alpha=0.5) for i in range(vals.shape[0])]
                vals = pt.loc[:,upar.parnme].values
                [ax.plot(dts,vals[i,:],"b",lw=0.1,alpha=0.5) for i in range(vals.shape[0])]
                ax.set_ylabel("annual ghb head")
                ax.set_title("annual ghb head layer {0}".format(uk+1),loc="left")
                ax.grid()
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)


        for op,t,lower,upper in zip(otherpar.parnme,otherpar.partrans,otherpar.parlbnd,otherpar.parubnd):
            if otherpar.loc[op,"partrans"] in ["fixed","tied"]:
                continue
            fig,axes = plt.subplots(2,1,figsize=(10,5))
            vals = pr.loc[:,op].values
            print("...",op)
            xvals = np.linspace(vals.min(),vals.max(),200)
            logtag = "log"
            if vals.min() <=0:
                logvals = vals
                logtag = "none"
                xvalslog = xvals
                
                
            else:
                logvals = np.log10(vals)
                xvalslog = np.linspace(logvals.min(),logvals.max(),200)
                 

            axes[0].hist(vals,bins=20,alpha=0.5,facecolor="0.5",density=True)
           
            axes[1].hist(logvals,bins=20,alpha=0.5,facecolor="0.5",density=True)
            
            vals = pt.loc[:,op].values
            
            kde = stats.gaussian_kde(vals)
            yvals = kde(xvals)

            if vals.min() <= 0 or logtag != "log":
                logvals = vals
                loglower = lower
                logupper = upper
                
                yvalslog = yvals
                
            else:
                logvals = np.log10(vals)
                loglower = np.log10(lower)
                logupper = np.log10(upper)
                
                kde = stats.gaussian_kde(logvals)
                yvalslog = kde(xvalslog)   
            
            axes[0].hist(vals,bins=20,alpha=0.5,facecolor="b",density=True)      
            axes[0].plot(xvals,yvals,"b-",lw=2.0)

            axes[1].hist(logvals,bins=20,alpha=0.5,facecolor="b",density=True)      
            axes[1].plot(xvalslog,yvalslog,"b-",lw=2.0)
            
            axes[0].set_title("par:{0}, transform:{1}".format(op,"none"),loc="left")
            axes[1].set_title("par:{0}, transform:{1}".format(op,logtag),loc="left")
            ylim = axes[0].get_ylim()
            axes[0].plot([upper,upper],ylim,"k--",lw=2.5)
            axes[0].plot([lower,lower],ylim,"k--",lw=2.5)
            ylim = axes[1].get_ylim()
            axes[1].plot([logupper,logupper],ylim,"k--",lw=2.5)
            axes[1].plot([loglower,loglower],ylim,"k--",lw=2.5)
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

    
def plot_model_input_summary(m_d,noptmax=None):
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    if noptmax is None:
        phidf = pd.read_csv(os.path.join(m_d, "pest.phi.actual.csv"))
        noptmax = phidf.iteration.max()
        # noptmax = "combined"
    pr = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.0.obs.jcb"))
    pt = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.{0}.obs.jcb".format(noptmax)))

    obs = pst.observation_data
    arrobs = obs.loc[obs.oname=="arrobs",:].copy()
    arrobs = arrobs.loc[arrobs.usecol=="pval"]
    arrobs.sort_index(inplace=True)
    listobs = obs.loc[obs.oname=="listobs",:].copy()
    listobs.sort_index(inplace=True)
    
    gobs = obs.loc[obs.oname=="ghbhead",:].copy()
    gobs["datetime"] = pd.to_datetime(gobs.datetime)
    gobs["k"] = gobs.k.astype(int)
    uks = gobs.k.unique()
    uks.sort()

    site_name = os.path.split(m_d)[0]
    raw_gw_df = pd.read_csv(os.path.join(site_name,"processed_data","{0}.ts_data.csv".format(site_name)),index_col=0,parse_dates=True)
    raw_gw_df = raw_gw_df.loc[pd.notna(raw_gw_df.Alt),:]
            

    with PdfPages(os.path.join(m_d,"input_summary_{0}.pdf".format(noptmax))) as pdf:
        fig,axes = plt.subplots(len(uks),1,figsize=(10,10))
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        for uk,ax in zip(uks,axes):
            uobs = gobs.loc[gobs.k==uk,:].copy()
            uobs.sort_values(by="datetime",inplace=True)
            dts = uobs.datetime
            vals = pr.loc[:,uobs.obsnme].values
            [ax.plot(dts,vals[i,:],"0.5",alpha=0.15,lw=0.15) for i in range(vals.shape[0])]
            
            vals = pt.loc[:,uobs.obsnme].values
            [ax.plot(dts,vals[i,:],"b",alpha=0.15,lw=0.15) for i in range(vals.shape[0])]
            if "base" in pr.index:
                ax.plot(dts,pr.loc["base",uobs.obsnme].values,"k",alpha=0.5,lw=2.0,zorder=10)
            if "base" in pt.index:
                ax.plot(dts,pt.loc["base",uobs.obsnme].values,"m",alpha=0.5,lw=2.0,zorder=10)
            kraw = raw_gw_df.loc[raw_gw_df.klayer==uk,:].copy()
            kraw.sort_index(inplace=True)
            ax.scatter(kraw.index.values,kraw.Alt.values,marker='o',s=10,c='k')
            ax.set_title("ghb head layer {0}".format(uk+1),loc="left")
            ax.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)

        for oname in listobs.obsnme:
            print("...",oname)
            #try:
            fig,axes = plt.subplots(2,1,figsize=(7,7))

            xvals = np.linspace(pr.loc[:,oname].values.min(),pr.loc[:,oname].values.max(),200)
            try:
                kde = stats.gaussian_kde(pt.loc[:,oname].values)
                yvals = kde(xvals)
            except:
                yvals = np.zeros_like(xvals)
                yvals[:] = np.nan

            if min(pr.loc[:,oname].values.min(),pr.loc[:,oname].values.min()) <=0.0:
                axes[1].hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="0.5",density=True)
                axes[1].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor="b",density=True)
                axes[1].plot(xvals,yvals,"b-",lw=2.0)
                axes[1].set_title(oname,loc="left")
                ylim = axes[1].get_ylim()
                if "base" in pr.index:
                    v = pr._df.loc["base",oname]
                    axes[1].plot([v,v],ylim,"0.5",alpha=0.5,lw=2.0,ls="--",zorder=10)
                if "base" in pt.index:
                    v = pt._df.loc["base",oname]
                    axes[1].plot([v,v],ylim,"b",alpha=0.5,lw=2.0,ls="--",zorder=10)

            else:
                axes[1].hist(np.log10(pr.loc[:,oname].values),bins=20,alpha=0.5,facecolor="0.5",density=True)
                axes[1].hist(np.log10(pt.loc[:,oname].values),bins=20,alpha=0.5,facecolor="b",density=True)
                xvalslog = np.linspace(np.log10(pr.loc[:,oname].values.min()),np.log10(pr.loc[:,oname].values.max()),200)
                try:
                    kde = stats.gaussian_kde(np.log10(pt.loc[:,oname].values))
                    yvalslog = kde(xvalslog)
                except:
                    yvalslog = np.zeros_like(xvals)
                    yvalslog[:] = np.nan
                axes[1].plot(xvalslog,yvalslog,"b-",lw=2.0)
                axes[1].set_title(oname+" (log10)",loc="left")
                ylim = axes[1].get_ylim()
                if "base" in pr.index:
                    v = np.log10(pr._df.loc["base",oname])
                    axes[1].plot([v,v],ylim,"0.5",alpha=0.5,lw=2.0,ls="--",zorder=10)
                if "base" in pt.index:
                    v = np.log10(pt._df.loc["base",oname])
                    axes[1].plot([v,v],ylim,"b",alpha=0.5,lw=2.0,ls="--",zorder=10)

            
                
        
            axes[0].hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="0.5",density=True)
            axes[0].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor="b",density=True)
            axes[0].plot(xvals,yvals,"b-",lw=2.0)
            ylim = axes[0].get_ylim()
            if "base" in pr.index:
                v = pr._df.loc["base",oname]
                axes[0].plot([v,v],ylim,"0.5",alpha=0.5,lw=2.0,ls="--",zorder=10)
            if "base" in pt.index:
                v = pt._df.loc["base",oname]
                axes[0].plot([v,v],ylim,"b",alpha=0.5,lw=2.0,ls="--",zorder=10)

            axes[0].set_title(oname,loc="left")
           
            #ylim = axes[0].get_ylim()
            #axes[0].plot([lower,lower],ylim,"k--",lw=2)
            #axes[0].plot([upper,upper],ylim,"k--",lw=2)
            axes[0].set_yticks([])
            axes[1].set_yticks([])
            
            
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
            # except:
            #     print("-----------------------------------------------------------------",
            #           f"{oname} plot didn't work",
            #           # "pt.loc[:,oname].values:",
            #           # pt.loc[:,oname].values,
            #           "-----------------------------------------------------------------")
    
        for oname in arrobs.obsnme:
            print("...",oname)
            lower = pr._df.loc[pr.index[0],oname.replace("pval","lower_bound")]
            upper = pr._df.loc[pr.index[0],oname.replace("pval","upper_bound")]
            if upper > 1e20:
                upper = np.nan
                lower = np.nan
            fig,axes = plt.subplots(2,1,figsize=(7,7))

            xvals = np.linspace(pr.loc[:,oname].values.min(),pr.loc[:,oname].values.max(),200)
            kde = stats.gaussian_kde(pt.loc[:,oname].values)
            yvals = kde(xvals)

            if min(pr.loc[:,oname].values.min(),pr.loc[:,oname].values.min(),lower) <=0.0:
                axes[1].hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="0.5",density=True)
                axes[1].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor="b",density=True)
                axes[1].plot(xvals,yvals,"b-",lw=2.0)
                axes[1].set_title(oname,loc="left")
                ylim = axes[1].get_ylim()
                if "base" in pr.index:
                    v = pr._df.loc["base",oname]
                    axes[1].plot([v,v],ylim,"0.5",alpha=0.5,lw=2.0,ls="--",zorder=10)
                if "base" in pt.index:
                    v = pt._df.loc["base",oname]
                    axes[1].plot([v,v],ylim,"b",alpha=0.5,lw=2.0,ls="--",zorder=10)

            else:
                loglower = np.log10(lower)
                logupper = np.log10(upper)
                axes[1].hist(np.log10(pr.loc[:,oname].values),bins=20,alpha=0.5,facecolor="0.5",density=True)
                axes[1].hist(np.log10(pt.loc[:,oname].values),bins=20,alpha=0.5,facecolor="b",density=True)
                xvalslog = np.linspace(np.log10(pr.loc[:,oname].values.min()),np.log10(pr.loc[:,oname].values.max()),200)
                kde = stats.gaussian_kde(np.log10(pt.loc[:,oname].values))
                yvalslog = kde(xvalslog)
                axes[1].plot(xvalslog,yvalslog,"b-",lw=2.0)
                axes[1].set_title(oname+" (log10)",loc="left")
                ylim = axes[1].get_ylim()
                axes[1].plot([loglower,loglower],ylim,"k--",lw=2)
                axes[1].plot([logupper,logupper],ylim,"k--",lw=2)
                ylim = axes[1].get_ylim()
                if "base" in pr.index:
                    v = np.log10(pr._df.loc["base",oname])
                    axes[1].plot([v,v],ylim,"0.5",alpha=0.5,lw=2.0,ls="--",zorder=10)
                if "base" in pt.index:
                    v = np.log10(pt._df.loc["base",oname])
                    axes[1].plot([v,v],ylim,"b",alpha=0.5,lw=2.0,ls="--",zorder=10)


                
        
            axes[0].hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="0.5",density=True)
            axes[0].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor="b",density=True)
            axes[0].plot(xvals,yvals,"b-",lw=2.0)

            axes[0].set_title(oname,loc="left")
           
            ylim = axes[0].get_ylim()
            axes[0].plot([lower,lower],ylim,"k--",lw=2)
            axes[0].plot([upper,upper],ylim,"k--",lw=2)

            ylim = axes[0].get_ylim()
            if "base" in pr.index:
                v = pr._df.loc["base",oname]
                axes[0].plot([v,v],ylim,"0.5",alpha=0.5,lw=2.0,ls="--",zorder=10)
            if "base" in pt.index:
                v = pt._df.loc["base",oname]
                axes[0].plot([v,v],ylim,"b",alpha=0.5,lw=2.0,ls="--",zorder=10)

            axes[0].set_yticks([])
            axes[1].set_yticks([])
            
            
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

def plot_model_input_summary_ht_compare(m_d,noptmax=None):
    
    assert os.path.exists(os.path.join(m_d+"_ht_max"))
    assert os.path.exists(os.path.join(m_d+"_ht_min"))
    
    colors = {'base':'blue','ht_max':'darkred','ht_min':'gold'}
    
    keyruns = {}
    for h_d,tag in zip([m_d,m_d+"_ht_max",m_d+"_ht_min"],['base','ht_max','ht_min']):
        pst = pyemu.Pst(os.path.join(h_d,"pest.pst"))
        phidf = pd.read_csv(os.path.join(h_d,"pest.phi.actual.csv"))
        if noptmax is None:
            noptmax = phidf.iteration.max()
        pr = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(h_d,"pest.0.obs.jcb"))
        pt = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(h_d,"pest.{0}.obs.jcb".format(noptmax)))


        obs = pst.observation_data
        cobs = obs.loc[obs.oname=="arrobs",:].copy()
        cobs = cobs.loc[cobs.usecol=="pval"]
        cobs.sort_index(inplace=True)
        gobs = obs.loc[obs.oname=="ghbhead",:].copy()
        gobs["datetime"] = pd.to_datetime(gobs.datetime)
        gobs["k"] = gobs.k.astype(int)
        uks = gobs.k.unique()
        uks.sort()
        
        keyruns[tag] = {"pr":pr,"pt":pt, "gobs":gobs, "cobs":cobs}

    with PdfPages(os.path.join(m_d,"input_summary_{0}_ht_compare.pdf".format(noptmax))) as pdf:
        fig,axes = plt.subplots(len(uks),1,figsize=(10,10))
        for uk,ax in zip(uks,axes):
            for tag in keyruns.keys():
                pr = keyruns[tag]["pr"]
                pt = keyruns[tag]["pt"]
                gobs = keyruns[tag]["gobs"]
                cobs = keyruns[tag]["cobs"]
                
                uobs = gobs.loc[gobs.k==uk,:].copy()
                uobs.sort_values(by="datetime",inplace=True)
                dts = uobs.datetime
                vals = pr.loc[:,uobs.obsnme].values
                [ax.plot(dts,vals[i,:],"0.5",alpha=0.5,lw=0.25) for i in range(vals.shape[0])]
                vals = pt.loc[:,uobs.obsnme].values
                [ax.plot(dts,vals[i,:],colors[tag],alpha=0.5,lw=0.25) for i in range(vals.shape[0])]
            # uobs = gobs.loc[gobs.k==uk,:].copy()
            # uobs.sort_values(by="datetime",inplace=True)
            # dts = uobs.datetime
            # vals = pr.loc[:,uobs.obsnme].values
            # [ax.plot(dts,vals[i,:],"0.5",alpha=0.5,lw=0.25) for i in range(vals.shape[0])]
            # vals = pt.loc[:,uobs.obsnme].values
            # [ax.plot(dts,vals[i,:],"b",alpha=0.5,lw=0.25) for i in range(vals.shape[0])]
            ax.set_title("ghb head layer {0}".format(uk+1),loc="left")
            ax.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)
    
        for oname in cobs.obsnme:
            fig,axes = plt.subplots(2,1,figsize=(7,7))
            for tag in keyruns.keys():
                pr = keyruns[tag]["pr"]
                pt = keyruns[tag]["pt"]
                gobs = keyruns[tag]["gobs"]
                cobs = keyruns[tag]["cobs"]
                lower = pr._df.loc[pr.index[0],oname.replace("pval","lower_bound")]
                upper = pr._df.loc[pr.index[0],oname.replace("pval","upper_bound")]
                if upper > 1e20:
                    upper = np.nan
                    lower = np.nan
                

                xvals = np.linspace(pr.loc[:,oname].values.min(),pr.loc[:,oname].values.max(),200)
                kde = stats.gaussian_kde(pt.loc[:,oname].values)
                yvals = kde(xvals)

                if min(pr.loc[:,oname].values.min(),pr.loc[:,oname].values.min(),lower) <=0.0:
                    axes[1].hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="0.5",density=True)
                    axes[1].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor=colors[tag],density=True)
                    axes[1].plot(xvals,yvals,colors[tag],lw=2.0)
                    axes[1].set_title(oname,loc="left")

                else:
                    loglower = np.log10(lower)
                    logupper = np.log10(upper)
                    axes[1].hist(np.log10(pr.loc[:,oname].values),bins=20,alpha=0.5,facecolor="0.5",density=True)
                    axes[1].hist(np.log10(pt.loc[:,oname].values),bins=20,alpha=0.5,facecolor=colors[tag],density=True)
                    xvalslog = np.linspace(np.log10(pr.loc[:,oname].values.min()),np.log10(pr.loc[:,oname].values.max()),200)
                    kde = stats.gaussian_kde(np.log10(pt.loc[:,oname].values))
                    yvalslog = kde(xvalslog)
                    axes[1].plot(xvalslog,yvalslog,colors[tag],lw=2.0)
                    axes[1].set_title(oname+" (log10)",loc="left")
                    ylim = axes[1].get_ylim()
                    axes[1].plot([loglower,loglower],ylim,"k--",lw=2)
                    axes[1].plot([logupper,logupper],ylim,"k--",lw=2)

                    
            
                axes[0].hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="0.5",density=True)
                axes[0].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor=colors[tag],density=True, label="pt "+tag)
                axes[0].plot(xvals,yvals,colors[tag],lw=2.0)

            axes[0].set_title(oname,loc="left")
        
            ylim = axes[0].get_ylim()
            axes[0].plot([lower,lower],ylim,"k--",lw=2)
            axes[0].plot([upper,upper],ylim,"k--",lw=2)
            axes[0].set_yticks([])
            axes[1].set_yticks([])
            axes[0].legend(loc='upper left')
            
            
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)


def plot_lstbud(m_d,noptmax=None):
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    if noptmax is None:
        phidf = pd.read_csv(os.path.join(m_d, "pest.phi.actual.csv"))
        noptmax = phidf.iteration.max()
        # noptmax = "combined"
    pr = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.0.obs.jcb"))
    pt = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.{0}.obs.jcb".format(noptmax)))

    obs = pst.observation_data
    cobs = obs.loc[obs.oname=="lstcum",:].copy()
    cobs["datetime"] = pd.to_datetime(cobs.datetime,format="%Y%m%d")
    usecols = cobs.usecol.unique()
    usecols.sort()

    with PdfPages(os.path.join(m_d,"lstbud_summary_{0}.pdf".format(noptmax))) as pdf:
        for usecol in usecols:
            uobs = cobs.loc[cobs.usecol==usecol,:].copy()
            uobs.sort_values(by="datetime",inplace=True)

            fig,axes = plt.subplots(2,1,figsize=(7,7))
            dts = uobs.datetime.values
            vals = pr.loc[:,uobs.obsnme].values
            [axes[0].plot(dts,vals[i,:],"0.5",alpha=0.5,lw=0.1) for i in range(vals.shape[0])]
            vals = pt.loc[:,uobs.obsnme].values
            [axes[0].plot(dts,vals[i,:],"b",alpha=0.5,lw=0.1) for i in range(vals.shape[0])]
            axes[0].set_title("{0} cumulative".format(usecol),loc="left")

            uobs["obsnme"] = uobs.obsnme.apply(lambda x: x.replace("cum","inc"))
            uobs.index = uobs.obsnme.values
            vals = pr.loc[:,uobs.obsnme].values
            [axes[1].plot(dts,vals[i,:],"0.5",alpha=0.5,lw=0.1) for i in range(vals.shape[0])]
            vals = pt.loc[:,uobs.obsnme].values
            [axes[1].plot(dts,vals[i,:],"b",alpha=0.5,lw=0.1) for i in range(vals.shape[0])]
            axes[1].set_title("{0} incremental".format(usecol),loc="left")

            if "percent" in usecol:
                axes[0].set_ylim(-5,5)
                axes[1].set_ylim(-5,5)
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

def _plot_thread(i,master_dir):
    # while True:
    print(f"Starting plot thread for iteration {i}")

    plot_en_compaction(m_d=master_dir,noptmax=i)
    plot_model_input_summary(master_dir,noptmax=i)
    plot_lstbud(m_d=master_dir,noptmax=i)
    plot_par_summary(master_dir,noptmax=i)
    #if i != "combined": # todo spencer check?
    #    plot_en_subsidence(m_d=master_dir,noptmax=i)
    try:
        plot_WL_SUB_2_panel(m_d=master_dir,noptmax=i)
    except Exception as e:
        print("error in plot 2 panel: {0}".format(str(e)))

    print(f"Completed iteration {i}")
    # except Exception as e:
    #     print(f"Error in iteration {i}: {e}")
    # return i

def run_parallel_plots(master_dir, iters):
    from multiprocessing import Pool
    with Pool(processes=min(5, len(iters))) as pool:
        results = []
        for i in iters:
            r = pool.apply_async(_plot_thread, args=(i, master_dir))
            results.append(r)
        for r in results:
            print(r.get())

def plot_all(master_dir,plot_all_iters=False):
    if plot_all_iters:
        plot_phi(m_d=master_dir)
        phidf = pd.read_csv(os.path.join(master_dir,"pest.phi.actual.csv"))
        iters = [i for i in range(1,phidf.iteration.max()+1)]
        make_combined_posterior(master_dir)
        iters.append("combined")

        run_parallel_plots(master_dir, iters)
        # from multiprocessing import Process, Manager, Pool
        #
        # processes = []
        # for i in iters:
        #     # create a new process for each iteration to run the function _plot_thread in parallel (to speed things up)
        #     p = Process(target=_plot_thread,args=(i,master_dir))
        #     p.start()
        #     processes.append(p)
        # # wait for all processes to finish before continuing
        # for p in processes:
        #     print(f"Joining process {p.pid}")
        #     p.join()#timeout=10)  # Wait until child process terminates. no timeout -> infinite hang possible
        #
        #
        # pool = Pool(processes=min(5,len(iters)))
        # results = []
        # for i in iters:
        #     r = pool.apply_async(_plot_thread,args=(i,master_dir))
        #     results.append(r)
        # for r in results:
        #     print(r.get())
        # pool.close()
        # pool.join()
    else:
        #try:
        plot_en_compaction(m_d=master_dir)
        plot_model_input_summary(master_dir)
        plot_lstbud(m_d=master_dir)
        plot_par_summary(master_dir)
        plot_phi(m_d=master_dir)
        plot_en_subsidence(m_d=master_dir)
        plot_WL_SUB_2_panel(m_d=master_dir)
        #except Exception as e:
        #    pass


def setup_diff_obs(t_d):
    b_d = os.getcwd()
    os.chdir(t_d)
    df = process_diff_obs()
    os.chdir(b_d)
    return df

def process_diff_obs():
    infile = "datetime_model.csub.obs.csv"
    outfile = "diff_" + infile
    df = pd.read_csv(infile,index_col=0,parse_dates=True)
    dts = df.index
    vals = df["sim-subsidence-ft"].values
    diff,dt1,dt2 = [],[],[]
    for i in range(vals.shape[0]-1):
        if dts[i].year < 2010:
            continue
        for ii in range(i+1,vals.shape[0]):
            diff.append(vals[ii] - vals[i])
            dt1.append(dts[i])
            dt2.append(dts[ii])
    df2 = pd.DataFrame({"dt1":dt1,"dt2":dt2,"diff":diff})
    df2.index.name = "diffnum"
    df2.to_csv(outfile)
    return df2



def prep_for_hypoth_test(ht_direction,org_t_d,org_m_d,new_t_d=None,post_noptmax=None):
    # make sure the dt1 is coherent with fixed forecast ghb pars in setup pst
    #ht_oname = "oname:subdiff_otype:lst_usecol:diff_dt1:2024-12-31_dt2:2053-12-23"
    ht_dt1_year = 2024#pd.to_datetime("12-31-2024")
    ht_dt2_year = 2053#pd.to_datetime("12-31-2053")
    ht_direction = ht_direction.lower()
    assert ht_direction in ["min","max"]


    

    if new_t_d is None:
        new_t_d = org_t_d+"_ht"
    if os.path.exists(new_t_d):
        shutil.rmtree(new_t_d)
    shutil.copytree(org_t_d,new_t_d)

    phidf = pd.read_csv(os.path.join(org_m_d,"pest.phi.actual.csv"))

    if post_noptmax is None:
        post_noptmax = phidf.iteration.max()

    pst = pyemu.Pst(os.path.join(org_m_d,"pest.pst"))
    obs = pst.observation_data
    sobs = obs.loc[obs.oname=="subdiff",:].copy() 
    sobs["dt1"] = pd.to_datetime(sobs.dt1)
    sobs["dt2"] = pd.to_datetime(sobs.dt2)

    sobs = sobs.loc[sobs.dt1.dt.year==ht_dt1_year,:]
    assert sobs.shape[0] > 0
    sobs = sobs.loc[sobs.dt2.dt.year==ht_dt2_year,:]
    assert sobs.shape[0] > 0

    sobs.sort_values(by=["dt1","dt2"],ascending=False,inplace=True)
    ht_oname = sobs.obsnme.iloc[0]
    print("ht obs name:",ht_oname)
      

    #assert ht_oname in pst.obs_names
    
    obs.loc[ht_oname,"weight"] = 1.0 # placeholder, adjusted at run time by phi factor file
    obs.loc[ht_oname,"obgnme"] = "htobs"

    pf_df = pd.read_csv(os.path.join(new_t_d,pst.pestpp_options["ies_phi_factor_file"]),index_col=0,header=None,names=["tag","prop"])
    pf_df.loc[:,"prop"] = (pf_df.prop.values * 0.5) / pf_df.prop.sum()
    pf_df.loc["htobs","prop"] = 0.5
    pf_df.to_csv(os.path.join(new_t_d,pst.pestpp_options["ies_phi_factor_file"]),header=False)

    #noise = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(org_m_d,"pest.obs+noise.jcb"))
    oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(org_m_d,"pest.{0}.obs.jcb".format(post_noptmax)))
    mn,mx = oe.loc[:,ht_oname].min(),oe.loc[:,ht_oname].max()
   
    #first_half_rnames = oe.index.values[:int(oe.shape[0]/2)]
    #second_half_rnames = oe.index.values[int(oe.shape[0]/2):]
    if ht_direction == "min":
        oe.loc[:,ht_oname] = 0.0 # try to have no delayed compaction
    # try to have twice as much delayed compaction or at least 1 ft
    elif ht_direction == "max":
        oe.loc[:,ht_oname] = oe.loc[:,ht_oname].apply(lambda x: x*2.0) 
    #oe.loc[:,ht_oname] = 0.0 # try to have no delayed compaction
    oe.to_binary(os.path.join(new_t_d,"ht_noise.jcb"))
    pst.pestpp_options["ies_obs_en"] = "ht_noise.jcb"


    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(org_m_d,"pest.0.par.jcb"))
    pe = pe.loc[oe.index,:]
    pe.to_binary(os.path.join(new_t_d,"ht_prior.jcb"))
    pst.pestpp_options["ies_par_en"] = "ht_prior.jcb"
    pst.pestpp_options["predictions"] = ht_oname

    pst.control_data.noptmax = -2
    pst.write(os.path.join(new_t_d,"pest.pst"),version=2)

    pyemu.os_utils.run("pestpp-ies pest.pst",cwd=new_t_d)


def plot_compare_ht(m_d1,m_d2,ht_oname=None):
    pst = pyemu.Pst(os.path.join(m_d2,"pest.pst"))
    phidf1 = pd.read_csv(os.path.join(m_d1,"pest.phi.actual.csv"))
    phidf2 = pd.read_csv(os.path.join(m_d2,"pest.phi.actual.csv"))
    if ht_oname is None:
        ht_oname = pst.pestpp_options["predictions"]
        #ht_oname = "oname:subdiff_otype:lst_usecol:diff_dt1:2023-12-31_dt2:2053-12-23"

    obs = pst.observation_data
    csub = obs.loc[obs.oname=="arrobs",:].copy()
    csub = csub.loc[csub.usecol=="pval"]
    assert csub.shape[0] > 0
    csub.sort_index(inplace=True)

    noise = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d2,"pest.obs+noise.jcb"))

    oe1 = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d1,"pest.{0}.obs.jcb".format(phidf1.iteration.max())))
    oe2 = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d2,"pest.{0}.obs.jcb".format(phidf2.iteration.max())))
    labels = ["original","extra crispy"]
    colors = ["c","m"]

    with PdfPages(os.path.join(m_d2,"ht_compare.pdf")) as pdf:
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        
        xvals = np.linspace(0,20,200)
        for oe,c,lab in zip([oe1,oe2],colors,labels):
            kde = stats.gaussian_kde(oe.loc[:,ht_oname].values)
            yvals =kde(xvals)
            ax.plot(xvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
            ax.fill_between(xvals,0,yvals,facecolor=c,alpha=0.25)
            
        try:
            kde = stats.gaussian_kde(noise.loc[:,ht_oname].values)
            yvals = kde(xvals)
            ax.plot(xvals,yvals,"r--",lw=3.0)
        except Exception as e:
            pass
        #ax.hist(noise.loc[:,ht_oname].values,color="r",alpha=0.5,bins=20,density=True,label="target values")
        ax.legend(loc="upper right")
        ax.set_xlabel(ht_oname)
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)  



        fig,axes = plt.subplots(2,1,figsize=(10,10))
        pv1 = oe1.phi_vector
        pv2 = oe2.phi_vector
        xvals = np.linspace(min(pv1.min(),pv2.min()),max(pv1.max(),pv2.max()),200)
        ax = axes[0]
        for pv,c,lab in zip([pv1,pv2],colors,labels):
            kde = stats.gaussian_kde(pv)
            yvals =kde(xvals)
            ax.plot(xvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
            ax.fill_between(xvals,0,yvals,facecolor=c,alpha=0.25)
            
        ax.legend(loc="upper right")
        ax.set_title("phi",loc="left")
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)  

        pv1 = np.log10(pv1)
        pv2 = np.log10(pv2)
        xvals = np.linspace(min(pv1.min(),pv2.min()),max(pv1.max(),pv2.max()),200)
        ax = axes[1]
        for pv,c,lab in zip([pv1,pv2],colors,labels):
            kde = stats.gaussian_kde(pv)
            yvals =kde(xvals)
            ax.plot(xvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
            ax.fill_between(xvals,0,yvals,facecolor=c,alpha=0.25)
            
        ax.legend(loc="upper right")
        ax.set_title("phi (log)",loc="left")
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)  


        for oname in csub.obsnme:
            fig,axes = plt.subplots(2,1,figsize=(10,10))   
            mn,mx = 1e300,-1e300
            for oe in [oe1,oe2]:
                mn = min(mn,oe.loc[:,oname].min())
                mx = max(mx,oe.loc[:,oname].max())
                #print("...",oname,oe.loc[:,oname].values,oe.loc[:,oname].max()) 
            #print(oname,mn,mx) 
            xvals = np.linspace(mn,mx,200)
            if mn <= 0:
                logxvals = xvals
            else:
                logxvals = np.linspace(np.log10(mn),np.log10(mx),200)

            for oe,m,lab,c in zip([oe1,oe2],[m_d1,m_d2],labels,colors):         
                ax = axes[0]            
                kde = stats.gaussian_kde(oe.loc[:,oname].values)
                yvals =kde(xvals)
                ax.plot(xvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
                ax.fill_between(xvals,0,yvals,facecolor=c,alpha=0.25)
            
                ax.legend(loc="upper right")
                ax.set_title(oname,loc="left")

                ax = axes[1]
                vals = oe.loc[:,oname].values
                if mn > 0.0:
                    vals = np.log10(vals)
                kde = stats.gaussian_kde(vals)
                yvals =kde(logxvals)
                ax.plot(logxvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
                ax.fill_between(logxvals,0,yvals,facecolor=c,alpha=0.25)
            
                ax.legend(loc="upper right")
                ax.set_title(oname+"(log)",loc="left")


            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)  

def plot_compare_both_ht(m_d,ht_oname=None):
    assert os.path.exists(os.path.join(m_d+"_ht_max"))
    assert os.path.exists(os.path.join(m_d+"_ht_min"))
    m_d1 = m_d+"_ht_max"
    m_d2 = m_d+"_ht_min"
    
    
    pst = pyemu.Pst(os.path.join(m_d1,"pest.pst"))
    phidf = pd.read_csv(os.path.join(m_d,"pest.phi.actual.csv"))
    phidf1 = pd.read_csv(os.path.join(m_d1,"pest.phi.actual.csv"))
    phidf2 = pd.read_csv(os.path.join(m_d2,"pest.phi.actual.csv"))
    if ht_oname is None:
        ht_oname = pst.pestpp_options["predictions"]
        #ht_oname = "oname:subdiff_otype:lst_usecol:diff_dt1:2023-12-31_dt2:2053-12-23"

    obs = pst.observation_data
    csub= obs.loc[obs.oname=="arrobs",:].copy()
    csub = obs.loc[obs.oname=="arrobs",:].copy()
    csub = csub.loc[csub.usecol=="pval"]
    assert csub.shape[0] > 0
    csub.sort_index(inplace=True)

    noise = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d2,"pest.obs+noise.jcb"))

    oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.{0}.obs.jcb".format(phidf.iteration.max())))
    oe1 = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d1,"pest.{0}.obs.jcb".format(phidf1.iteration.max())))
    oe2 = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d2,"pest.{0}.obs.jcb".format(phidf2.iteration.max())))
    labels = ["original","ht max", "ht min"]
    colors = ["c","m","y"]

    with PdfPages(os.path.join(m_d,"ht_compare_both.pdf")) as pdf:
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        
        xvals = np.linspace(0,20,200)
        for oei,c,lab in zip([oe,oe1,oe2],colors,labels):
            kde = stats.gaussian_kde(oei.loc[:,ht_oname].values)
            yvals =kde(xvals)
            ax.plot(xvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
            ax.fill_between(xvals,0,yvals,facecolor=c,alpha=0.25)
            
        try:
            kde = stats.gaussian_kde(noise.loc[:,ht_oname].values)
            yvals = kde(xvals)
            ax.plot(xvals,yvals,"r--",lw=3.0)
        except Exception as e:
            pass
        #ax.hist(noise.loc[:,ht_oname].values,color="r",alpha=0.5,bins=20,density=True,label="target values")
        ax.legend(loc="upper right")
        ax.set_xlabel(ht_oname)
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)  



        fig,axes = plt.subplots(2,1,figsize=(10,10))
        pv = oe.phi_vector
        pv1 = oe1.phi_vector
        pv2 = oe2.phi_vector
        xvals = np.linspace(min(pv.min(),pv1.min(),pv2.min()),max(pv.max(),pv1.max(),pv2.max()),200)
        ax = axes[0]
        for pvi,c,lab in zip([pv,pv1,pv2],colors,labels):
            kde = stats.gaussian_kde(pvi)
            yvals =kde(xvals)
            ax.plot(xvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
            ax.fill_between(xvals,0,yvals,facecolor=c,alpha=0.25)
            
        ax.legend(loc="upper right")
        ax.set_title("phi",loc="left")
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)  

        pv = np.log10(pv)
        pv1 = np.log10(pv1)
        pv2 = np.log10(pv2)
        xvals = np.linspace(min(pv1.min(),pv2.min()),max(pv1.max(),pv2.max()),200)
        ax = axes[1]
        for pvi,c,lab in zip([pv,pv1,pv2],colors,labels):
            kde = stats.gaussian_kde(pvi)
            yvals =kde(xvals)
            ax.plot(xvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
            ax.fill_between(xvals,0,yvals,facecolor=c,alpha=0.25)
            
        ax.legend(loc="upper right")
        ax.set_title("phi (log)",loc="left")
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)  


        for oname in csub.obsnme:
            fig,axes = plt.subplots(2,1,figsize=(10,10))   
            mn,mx = 1e300,-1e300
            for oei in [oe,oe1,oe2]:
                mn = min(mn,oei.loc[:,oname].min())
                mx = max(mx,oei.loc[:,oname].max())
                #print("...",oname,oe.loc[:,oname].values,oe.loc[:,oname].max()) 
            #print(oname,mn,mx) 
            xvals = np.linspace(mn,mx,200)
            if mn <= 0:
                logxvals = xvals
            else:
                logxvals = np.linspace(np.log10(mn),np.log10(mx),200)

            for oei,m,lab,c in zip([oe,oe1,oe2],[m_d,m_d1,m_d2],labels,colors):         
                ax = axes[0]            
                kde = stats.gaussian_kde(oei.loc[:,oname].values)
                yvals =kde(xvals)
                ax.plot(xvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
                ax.fill_between(xvals,0,yvals,facecolor=c,alpha=0.25)
            
                ax.legend(loc="upper right")
                ax.set_title(oname,loc="left")

                ax = axes[1]
                vals = oei.loc[:,oname].values
                if mn > 0.0:
                    vals = np.log10(vals)
                kde = stats.gaussian_kde(vals)
                yvals =kde(logxvals)
                ax.plot(logxvals,yvals,color=c,lw=3.0,label="{0}".format(lab))
                ax.fill_between(logxvals,0,yvals,facecolor=c,alpha=0.25)
            
                ax.legend(loc="upper right")
                ax.set_title(oname+"(log)",loc="left")


            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)  

def test_csv_to_ghb(t_d):
    df = test_ghbhead_to_csv(t_d)
    df.to_csv(os.path.join(t_d,"input_ghb.csv"))
    b_d = os.getcwd()
    os.chdir(t_d)
    df = csv_to_ghb()
    os.chdir(b_d)
    return df

def csv_to_ghb():
    df = pd.read_csv("input_ghb.csv")
    df["row"] = 1
    df["col"] = 1
    df["lay"] = df.k + 1

    fnames = df.fname.unique()
    fnames.sort()
    for fname in fnames:
        fdf = df.loc[df.fname==fname,:]
        fdf.loc[:,["lay","row","col","bhead","cond"]].to_csv(fname,sep=" ",header=False,index=False)


def run_scenarios(location,t_d,m_d,post_itr=None,new_m_d=None,
                  new_t_d=None,num_workers=10,port=4003,
                  subset_size=None,usecondor=False,scenario_tag="",
                  run_prior_scenarios=False,scenariofile=""):
    # sys.path.insert(0,location)
    # import prep_data
    # reload(prep_data)
    # assert location in prep_data.__file__
    #
    # # super hack...
    # b_d = os.getcwd()
    # os.chdir(location)
    # # if rolling_window is not None:
    # #     prep_data.prep_data(use_delay,rolling_window)
    # # else:
    # prep_data.prep_scenario_csv(location, scenariofile)
    # os.chdir(b_d)

    scen_df = pd.read_csv(os.path.join(location,"processed_data","{0}.scenarios.csv".format(location)),index_col=0,parse_dates=True)
    scen_df = scen_df.dropna(axis=0)
    scen_df = scen_df.dropna(axis=1)
    assert len(scen_df) > 0
    scenario_names = list(set([c.split("_k")[0] for c in scen_df.columns]))
    scenario_names.sort()
    print(scenario_names)

    scenario_kvals = list(set([int(c.split("_k:")[1]) for c in scen_df.columns]))
    scenario_kvals.sort()
    if location.lower() == "h201" and 2 in scenario_kvals:
        scenario_kvals.remove(2)
    phidf = pd.read_csv(os.path.join(m_d,"pest.phi.actual.csv"))
    if post_itr is None:
        post_itr = phidf.iteration.max()
        # make_combined_posterior(m_d)
        # post_itr = "combined"

    print("using posterior iteration",post_itr)

    print("loading pst...")

    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    par = pst.parameter_data

    print("mapping scenarios into parameter data...")
    dpar = par.loc[par.pname == "directghbhead", :].copy()
    assert (dpar.shape[0] > 0)
    dpar["datetime"] = pd.to_datetime(dpar.datetime)
    dpar["k"] = dpar.k.astype(int) + 1 #for the one-based substraction in the par naming
    dtset = set(scen_df.index.tolist())
    scenario_start_datetime = pd.to_datetime("1-feb-2024")
    dpar_ks = dpar['k'].unique().tolist()
    scen_df = scen_df.loc[:, [col for col in scen_df.columns if any("_k:{0}".format(keyword) in col for keyword in dpar_ks)]]
    scenario_kvals = list(set([int(c.split("_k:")[1]) for c in scen_df.columns]))
    for name in scenario_names:
        print("...",name)
        sscen_df = scen_df.loc[:,[c for c in scen_df.columns if name in c]]
        assert sscen_df.shape[0] >= len(dtset)
        dpar[name] = np.nan
        for k in scenario_kvals:
        # for k in dpar_ks:
            print("...",name,k)
            dkpar = dpar.loc[dpar.k==k,:]
            assert dkpar.shape[0] > 0
            kstr = "_k:{0}".format(k)
            ssscen_df = sscen_df.loc[:,[c for c in sscen_df.columns if kstr in c]]
            assert ssscen_df.shape[0] == len(dtset)
            assert ssscen_df.shape[1] >= 1
            sname = ssscen_df.columns[0]
            for pname,dt in zip(dkpar.parnme,dkpar.datetime):
                if dt < scenario_start_datetime:
                    continue
                if dt not in dtset:
                    print("missing",dt,"interpolating...")
                    scen_df["dtdiff"] = np.abs((scen_df.index - dt).days)
                    nearest = scen_df.dtdiff.idxmin()
                    print("...interpolated from ",dt," to ",nearest)
                    dpar.loc[pname, name] = ssscen_df.loc[nearest, sname]

                else:
                    dpar.loc[pname,name] = ssscen_df.loc[dt,sname]
                    print("...",name,k,dt)
    #print(dpar.columns)

    print("...loading ensembles")
    pr_pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.0.par.jcb"))
    pt_pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.{0}.par.jcb".format(post_itr)))

    if subset_size is not None:
        idxs = pr_pe.index[-1*subset_size:].tolist()
        if "base" in pr_pe.index and "base" not in idxs:
            idxs[-1] = "base"
        pr_pe = pr_pe.loc[idxs,:]
        idxs = pt_pe.index[-1 * subset_size:].tolist()
        if "base" in pt_pe.index and "base" not in idxs:
            idxs[-1] = "base"
        pt_pe = pt_pe.loc[idxs, :]


    dset = set(dpar.parnme.tolist())
    dpar_in_pe = [name for name in pr_pe.columns if name in dset]
    #print(pr_pe.loc[:,dpar_in_pe])
    assert len(dpar_in_pe) == dpar.shape[0]

    # print("forming combo pr-pt ensemble")
    # combo_pe = pr_pe._df.copy()
    # combo_pe.index = ["bayes:pr_real:{0}_scen:base".format(i) for i in pr_pe.index]
    t = pt_pe._df.copy()
    t.index = ["bayes:pt_real:{0}_scen:base".format(i) for i in pt_pe.index]
    #combo_pe = pd.concat([combo_pe,t])
    combo_pe = t.copy()
    t = None
    t = pr_pe._df.copy()

    t.index = ["bayes:pr_real:{0}_scen:base".format(i) for i in pr_pe.index]
    combo_pe_pr = t.copy()
    t = None
    scen_ens = [combo_pe.copy()]
    scen_ens_pr = [combo_pe_pr.copy()]
    for sname in scenario_names:
        print("...processing ensembles for scenario",name)
        sdpar = dpar.loc[:,sname].dropna()
        assert sdpar.shape[0] > 0,str(dpar.columns)
        scen_pe = combo_pe.copy()

        scen_pe.index = [i.replace("scen:base","scen:{0}".format(sname)) for i in scen_pe.index]
        print(scen_pe.index)
        for name,val in zip(sdpar.index,sdpar.values):
            scen_pe.loc[:,name] = val
        scen_ens.append(scen_pe)

        scen_pe = combo_pe_pr.copy()
        scen_pe.index = [i.replace("scen:base","scen:{0}".format(sname)) for i in scen_pe.index]
        #print(scen_pe.index)
        for name,val in zip(sdpar.index,sdpar.values):
            scen_pe.loc[:,name] = val
        scen_ens_pr.append(scen_pe)

    print("...concat scenario ensembles")
    scen_en = pd.concat(scen_ens)
    print(scen_en.shape)
    scen_en_pr = pd.concat(scen_ens_pr)
    print(scen_en_pr.shape)

    print("...prep dirs and save ensemble")
    if new_t_d is None:
        new_t_d = t_d + "_scenarios"
    if os.path.exists(new_t_d):
        shutil.rmtree(new_t_d)
    shutil.copytree(t_d,new_t_d)

    #the name "scenarios.jcb" is hard-coded in the morris run function
    pyemu.ParameterEnsemble(pst=pst,df=scen_en).to_binary(os.path.join(new_t_d,"scenarios.jcb"))
    pyemu.ParameterEnsemble(pst=pst,df=scen_en_pr).to_binary(os.path.join(new_t_d,"scenarios_pr.jcb"))

    if new_m_d is None:
        new_m_d = m_d
        if len(scenario_tag) > 0:
            new_m_d += scenario_tag
        else:
            new_m_d += "_scenarios"

    if os.path.exists(new_m_d):
        shutil.rmtree(new_m_d)
    pst.pestpp_options["ies_par_en"] = "scenarios.jcb"
    pst.pestpp_options.pop("ies_obs_en",None)
    pst.pestpp_options.pop("ies_num_reals",None)
    pst.pestpp_options["ies_include_base"] = False
    pst.pestpp_options.pop("ies_bad_phi_sigma",None)
    
    for name in scenario_names:
        par.loc[dpar.parnme,name] = dpar[name].values
    pst.control_data.noptmax = -1
    print("...saving pst")
    pst.write(os.path.join(new_t_d,"pest.pst"),version=2)

    if usecondor:
        import write_condor
        write_condor.run_condor(template_ws=new_t_d,master_dir=new_m_d,num_workers=num_workers,pestpp="ies")
    else:
        run_local(worker_dir=new_t_d,master_dir=new_m_d,num_workers=num_workers,pestpp="ies",port=port)

    if run_prior_scenarios:
        pst.pestpp_options["ies_par_en"] = "scenarios_pr.jcb"
        pst.write(os.path.join(new_t_d,"pest.pst"),version=2)

        if usecondor:
            import write_condor
            write_condor.run_condor(template_ws=new_t_d,master_dir=new_m_d+"_prior",num_workers=num_workers,pestpp="ies")
        else:
            run_local(worker_dir=new_t_d,master_dir=new_m_d+"_prior",num_workers=num_workers,pestpp="ies",port=port)

def plot_en_compaction_scenarios(m_d,bayes_stance="pt"):

    pst = pyemu.Pst(os.path.join(m_d, "pest.pst"))
    pt_en = pyemu.ObservationEnsemble.from_binary(None,
                                                  os.path.join(m_d,
                                                               'pest.0.obs.jcb'))
    print(pt_en.index)
    scenario_names = list(set([i.split("scen:")[1].split("_")[0] for i in pt_en.index]))
    print(scenario_names)
    #exit()
    #scenario_colors = ["m","c","g","b","r"]
    scenario_colors = cm.jet(np.linspace(0, 1, len(scenario_names)))
    scenario_index_dict = {}
    for scenario_name in scenario_names:
        idxs = [i for i in pt_en.index if "_scen:"+scenario_name in i and "bayes:{0}".format(bayes_stance) in i]
        assert len(idxs) > 0
        scenario_index_dict[scenario_name] = idxs

    obs = pst.observation_data
    sobs = obs.loc[obs.usecol == "sim-subsidence-ft", :].copy()
    assert sobs.shape[0] > 0
    sobs["datetime"] = pd.to_datetime(sobs.datetime, format="%Y%m%d")

    obsobs = sobs.loc[sobs.observed == True, :].copy()
    nzobs = sobs.loc[sobs.weight > 0, :].copy()

    nzobs.sort_values(by="datetime", inplace=True)
    obsobs.sort_values(by="datetime", inplace=True)
    base_dict = {}
    with PdfPages(os.path.join(m_d, "compaction_summary_scenarios.pdf")) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))

        ymax = obsobs.obsval.max() * 2  # ,
        # pt_en.loc[:, sobs.obsnme].values.max())  # np.round((nzobs.obsval.max() + 10.)/10.) * 10.
        ymin = min(obsobs.obsval.min(), 0)

        ax = axes[0]

        for name,color in zip(scenario_names,scenario_colors):

            dts = sobs.datetime.values
            vals = pt_en.loc[scenario_index_dict[name], sobs.obsnme].values
            [ax.plot(dts, vals[i, :], color=color, lw=0.5, alpha=0.5,
                     label=name) for i in range(vals.shape[0])]
            base_real = [i for i in scenario_index_dict[name] if "_real:base" in i]

            if len(base_real) == 1:
                #axes[0].plot(dts, pt_en.loc["base", sobs.obsnme].values, color='b', lw=2.5,
                #             label='base posterior')
                axes[1].plot(dts, pt_en.loc[base_real[0], sobs.obsnme].values, color=color, lw=2.5,
                             label=name)
                base_dict[name] = pd.Series(data=pt_en.loc[base_real[0], sobs.obsnme].values,index=dts)

            for i, mtype in enumerate(obsobs.source.unique()):
                tmp = nzobs.loc[nzobs.source == mtype, :].copy()
                tmp = tmp.loc[tmp.weight > 0, :]
                # if tmp.shape[0] > 0:
                tmp.sort_values(by="datetime", inplace=True)
                axes[0].scatter(tmp.datetime.values, tmp.obsval.values, marker="o", c='r', s=4,
                                label="observed", zorder=10)
                axes[1].scatter(tmp.datetime.values, tmp.obsval.values, marker="o", c='r', s=6,
                                label="observed", zorder=10)
            base_df = pd.DataFrame(base_dict)
            base_df.to_csv(os.path.join(m_d,"base_scenarios.csv"))



        axes[0].set_title('ensemble total compaction obs vs sim', loc="left")
        axes[1].set_title('base real total compaction obs vs sim', loc="left")

        xlim = axes[0].get_xlim()

        for iax, ax in enumerate(axes):
            handles, labels = ax.get_legend_handles_labels()
            handelz, labelz = [], []
            for i, l in enumerate(labels):
                if l not in labelz:
                    labelz.append(l)
                    handelz.append(handles[i])
            ax.legend(handles=handelz, labels=labelz)
            ax.set_xlim(xlim)
            ax.grid()
            if iax < 2:
                ax.set_ylim(ymin, ymax)  # ylim)

        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)


def plot_en_subsidence_scenarios(site_name, m_d, new_m_d,
                                 plot_obs=False,
                                 noptmax=None,
                                 ):
    """
    Parameters
    ----------
    plot_obs: bool, optional
        Flag to specify to plot observed subsidence or not. Default is False

    Returns
    -------
    None.
    Saves plot with scenario subsidence + ensemble range and base subsidence
    """

    if not isinstance(plot_obs, bool):
        raise TypeError("The 'plot_obs' parameter must be a boolean.")

    print("Creating subsidence ensemble figure")

    # Get list of scenario names (not including base)
    scenario_names = [s for s in pd.read_csv(os.path.join(new_m_d, "base_scenarios.csv"), index_col=0).columns.tolist() if s != "base"]
    print(scenario_names)
    missing = [s for s in scenario_names if s not in label_keys.keys()]
    print("MISSING:",missing)
    
    #assert all(s in label_keys.keys() for s in scenario_names), "All scenario keys need to be added to label_keys dict"
    for m in missing:
        label_keys[m] = m
    # Load observed subsidence data
    sub_file = [s for s in os.listdir(m_d) if "sub_data" in s][0]
    sub_data = pd.read_csv(os.path.join(m_d, sub_file),
                           index_col=0,
                           parse_dates=True)

    # ---------- Load the calibration PEST results

    pst = pyemu.Pst(os.path.join(m_d, "pest.pst"))
    phidf = pd.read_csv(os.path.join(m_d, "pest.phi.actual.csv"))
    if noptmax is None:
        noptmax = phidf.iteration.max()
        # noptmax = "combined"

    jcb_path = os.path.join(m_d, f'pest.{noptmax}.obs.jcb')
    if not os.path.exists(jcb_path):
        raise FileNotFoundError(f".jcb file does not exist for baseline: {jcb_path}")
    pt_en = pyemu.ObservationEnsemble.from_binary(None,
                                                  jcb_path)

    obs = pst.observation_data
    # base pest files all relate to base scenario, so grab all indices
    baselie_index_dict = {}  # Get indices of scenario outputs
    idxs = [i for i in pt_en.index]
    assert len(idxs) > 0
    baselie_index_dict['base'] = idxs

    sobs = obs.loc[obs.usecol == "sim-subsidence-ft", :].copy()
    assert sobs.shape[0] > 0
    sobs["datetime"] = pd.to_datetime(sobs.datetime, format="%Y%m%d")
    dts = sobs.datetime.values  # date time stamps
    pt_sc_base = pt_en.loc[baselie_index_dict['base'], sobs.obsnme]._df  # get scenario and obs

    # Grab data within focus period for right-hand plot panel
    low = np.searchsorted(pd.to_datetime(dts), pd.to_datetime('01/01/2020'))
    high = np.searchsorted(pd.to_datetime(dts), pd.to_datetime('01/01/2040'))
    dts_slice = dts[(dts < pd.to_datetime('01/01/2040')) & (dts >= pd.to_datetime('01/01/2020'))]

    # ---------- Load the scenario PEST results

    jcb_path_scen = os.path.join(new_m_d, 'pest.0.obs.jcb')
    if not os.path.exists(jcb_path_scen):
        raise FileNotFoundError(f".jcb file does not exist for scenario: {jcb_path_scen}")
    pt_en_scen = pyemu.ObservationEnsemble.from_binary(None,
                                                  jcb_path_scen)

    # Grab the data for specified scenario names
    scenario_index_dict = {}  # Get indices of scenario outputs
    for scenario_name in scenario_names:
        idxs = [i for i in pt_en_scen.index if "_scen:" + scenario_name in i and "bayes:pt" in i]
        # idxs = [i for i in pt_en_scen.index]
        assert len(idxs) > 0
        scenario_index_dict[scenario_name] = idxs

    # ---------- Plot ensemble of model results --> One plot for each scenario

    for scenario_name in scenario_names:
        # Initialize the figure
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[1, 0.5])
        fig.subplots_adjust(wspace=0)

        # Grab scenario data
        pt_sc_ob = pt_en_scen.loc[scenario_index_dict[scenario_name], sobs.obsnme]._df  # get scenario and obs

        # Plot scenario results
        ax[0].plot(dts,
                   pt_sc_ob[pt_sc_ob.index.str.contains('base_scen')].values.flatten(),
                   color='grey',
                   lw=2,
                   label='Optimization Scenario',
                   zorder=1)
        ax[1].plot(dts_slice,
                   pt_sc_ob[pt_sc_ob.index.str.contains('base_scen')].values.flatten()[low:high],
                   color='grey',
                   lw=2,
                   label='Optimization Scenario',
                   zorder=1)

        # Plot calibration model base results
        ax[0].plot(dts,
                   pt_sc_base[pt_sc_base.index.str.contains('base')].values.flatten(),
                   color='tab:blue',
                   lw=2,
                   label='Calibrated Run',
                   ls='--')
        ax[1].plot(dts_slice,
                   pt_sc_base[pt_sc_base.index.str.contains('base')].values.flatten()[low:high],
                   color='tab:blue',
                   lw=2,
                   label='Calibrated Run',
                   ls='--')

        # Plot range of scenario results
        ax[0].fill_between(dts,
                           pt_sc_ob[~pt_sc_ob.index.str.contains('base_scen')].min(axis=0).values,
                           pt_sc_ob[~pt_sc_ob.index.str.contains('base_scen')].max(axis=0).values,
                           facecolor='grey',
                           alpha=0.25,
                           zorder=1)
        ax[1].fill_between(dts_slice,
                           pt_sc_ob[~pt_sc_ob.index.str.contains('base_scen')].min(axis=0).values[low:high],
                           pt_sc_ob[~pt_sc_ob.index.str.contains('base_scen')].max(axis=0).values[low:high],
                           facecolor='grey',
                           alpha=0.25,
                           zorder=1)

        # Optionally plot observed subsidence
        if plot_obs:
            sub_data.reset_index().plot(ax=ax[0],
                                        x='Date',
                                        y='Subsidence_ft',
                                        c='k',
                                        markersize=3,
                                        label='Observed',
                                        zorder=10,
                                        ls='',
                                        marker='o',
                                        legend=False)

        # Plot formatting
        fig.suptitle(f'{site_name}, {label_keys[scenario_name]} Ensemble Results', fontsize=13)
        ax[0].set_xlim([sub_data.index.min(), pd.to_datetime('10/01/2040')])
        ax[0].legend(loc=3, framealpha=1, fontsize=10)
        ax[0].grid(True)
        ax[0].set_xlabel('')
        ax[0].set_ylim([ax[0].get_ylim()[1], ax[0].get_ylim()[0]])
        ax[0].set_ylabel('Subsidence (ft)', fontsize=12)
        
        ax[1].grid(True)
        ax[1].set_xlim([pd.to_datetime('10/01/2020'), pd.to_datetime('10/01/2040')])
        ax[1].set_ylim([ax[1].get_ylim()[1], ax[1].get_ylim()[0]])
        ax[1].set_ylabel('')
        ax[1].legend().remove()
        
        xticks = ax[1].get_xticklabels()
        xticks = [x if i % 2 == 0 else '' for i, x in enumerate(xticks)]
        ax[1].set_xticklabels(xticks, fontsize=10)
        ax[0].tick_params(axis='x', labelsize=10)
        ax[0].tick_params(axis='y', labelsize=10)
        ax[1].tick_params(axis='x', labelsize=10)
        ax[1].tick_params(axis='y', labelsize=10)
        
        fig.tight_layout()
        
        fig.supxlabel('Year', y=-0.02, fontsize=12)
        
        savePath = os.path.join(new_m_d, "figures")
        os.makedirs(savePath, exist_ok=True)
        
        plt.savefig(os.path.join(savePath, f"{site_name}_subsidence_{scenario_name}_{noptmax}.png"),
                    dpi=250,
                    bbox_inches='tight')
        
        plt.close(fig)

def plot_en_subsidence(m_d,
                       noptmax=None
                       ):
    """
    Parameters
    ----------
    Returns
    -------
    """

    print("Creating subsidence ensemble figure")

    # Load observed subsidence and aquifer water levels
    sub_file = [s for s in os.listdir(m_d) if "sub_data" in s][0]
    sub_observed = pd.read_csv(os.path.join(m_d, sub_file),
                               index_col=0,
                               parse_dates=True)

    ts_file = [s for s in os.listdir(os.path.join(m_d, "processed_data")) if "ts_data" in s][0]
    wl_observed = pd.read_csv(os.path.join(m_d, "processed_data", ts_file),
                              index_col=0,
                              parse_dates=True)

    # ---------- Load the calibration PEST results

    pst = pyemu.Pst(os.path.join(m_d, "pest.pst"))
    phidf = pd.read_csv(os.path.join(m_d, "pest.phi.actual.csv"))
    if noptmax is None:
        noptmax = phidf.iteration.max()
        # noptmax = "combined"

    jcb_path = os.path.join(m_d, f'pest.{noptmax}.obs.jcb')
    if not os.path.exists(jcb_path):
        raise FileNotFoundError(f".jcb file does not exist for baseline: {jcb_path}")
    pt_en = pyemu.ObservationEnsemble.from_binary(None,
                                                  jcb_path)

    obs = pst.observation_data
    # Changing to grab all the index values since they don't appear to be labeled...
    baselie_index_dict = {}  # Get indices of scenario outputs
    idxs = [i for i in pt_en.index]
    assert len(idxs) > 0
    baselie_index_dict['base'] = idxs

    sobs = obs.loc[obs.usecol == "sim-subsidence-ft", :].copy()
    assert sobs.shape[0] > 0
    sobs["datetime"] = pd.to_datetime(sobs.datetime, format="%Y%m%d")
    dts = sobs.datetime.values  # date time stamps
    pt_sc_ob = pt_en.loc[baselie_index_dict['base'], sobs.obsnme]._df

    # ---------- Plot the base results, PEST ensemble, observed WL, and observed sub

    # Initialize the figure
    fig,ax = plt.subplots(figsize=(9,4))

    # Plot base realization
    ax.plot(dts,
            pt_sc_ob[pt_sc_ob.index.str.contains('base')].values.flatten(),
            color='tab:blue',
            lw=2,
            label='Calibrated Run')

    # Plot ensemble range
    ax.fill_between(dts,
                    pt_sc_ob[~pt_sc_ob.index.str.contains('base')].min(axis=0).values,
                    pt_sc_ob[~pt_sc_ob.index.str.contains('base')].max(axis=0).values,
                    facecolor='tab:blue',
                    alpha=0.35)

    # Plot observed subsidence
    sub_observed.reset_index().plot(ax=ax,
                                    x='Date',
                                    y='Subsidence_ft',
                                    c='k',
                                    markersize=3,
                                    label='Observed',
                                    zorder=10,
                                    ls='--',
                                    marker='o',
                                    legend=False
                                    )

    # Plot WL on secondary axis
    ax2 = ax.twinx()
    colors = ['tab:orange', 'tab:green', 'tab:purple']
    aquifers = wl_observed['Aquifer'].unique()
    site_name = os.path.normpath(m_d).split(os.path.sep)[0]
    if site_name == 'Q288':
        aquifers = ['Lower', 'Santa Margarita']
    for i, aquifer in enumerate(aquifers):
        wl_observed.loc[wl_observed['Aquifer'] == aquifer, 'interpolated'].plot(ax=ax2,
                                                                                color=colors[i],
                                                                                label=f'{aquifer} Aquifer WL',
                                                                                lw=1)
    # Plot formatting
    ax.set_title(f'{site_name}, Calibrated Run and Observed', fontsize=12)
    ax.set_xlim([sub_observed.index.min(), pd.to_datetime('10/01/2025')])
    h, l = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=h + h2,
               labels=l + l2,
               loc=3,
               framealpha=1,
               fontsize=9)
    ax2.set_ylabel('Water Level (ft)', fontsize=11)
    ax.grid(True)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Subsidence (ft)', fontsize=11)
    ax.set_ylim([ax.get_ylim()[1], ax.get_ylim()[0]])
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax2.tick_params(axis='x', labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    
    fig.tight_layout()
    savePath = os.path.join(m_d, "figures")
    os.makedirs(savePath, exist_ok=True)
    
    plt.savefig(os.path.join(savePath, f"{site_name}_subsidence_{noptmax}.png"),
                dpi=250,
                bbox_inches='tight')
    plt.close(fig)

def plot_scenarios(site_name, 
                   m_d
                   ):
    """
    Parameters
    Returns
    -------
    None.
    """

    print("Creating model scenario figure...")

    # Build subsidence path and load data
    sub_file = [s for s in os.listdir(m_d) if "sub_data" in s][0]
    sub_observed = pd.read_csv(os.path.join(m_d, sub_file),
                               index_col=0,
                               parse_dates=True)

    # Look for directories matching the "m_d_scenarioname" pattern
    scen_dirs = [subdir for subdir in next(os.walk(site_name))[1] if subdir.startswith(os.path.basename(m_d))]
    scen_dirs = [s for s in scen_dirs if len(s.split(os.path.basename(m_d))[-1]) > 0]

    # Build and check scenario paths -> automatically finds all scenarios
    scen_paths = {}
    scenario_plot_keys = {}
    for scen_dir in scen_dirs:
        # Construct the path to the base_scenarios.csv file
        path = os.path.join(site_name, scen_dir, "base_scenarios.csv")
        assert os.path.exists(path)
        # Get name scenario of run
        scenario_name = scen_dir.split(os.path.basename(m_d))[-1][1:]
        scen_paths[scenario_name] = path
        # For this scenario run, get list of scenario names (not including base)
        scenario_names = [s for s in pd.read_csv(path, index_col=0).columns.tolist() if s != "base"]
        assert len(scenario_names) > 0
        scenario_plot_keys[scenario_name] = scenario_names
    assert len(scen_dirs) > 0

    # scenario_plot_keys: Maybe also a user input? That way we can specify exactly which scenarios are plotted

    # Plot the scenario results
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[1, 0.5])
    fig.subplots_adjust(wspace=0)
    for k, scen in enumerate(scen_paths.keys()):
        # Load scenario data
        dat = pd.read_csv(scen_paths[scen],
                          index_col=0,
                          parse_dates=True)

        # Catches a naming inconsistency
        if 'DWR_scenarios' in scen and 'mt-glidepath-not-below-mt' not in dat.keys():
            try:
                dat['mt-glidepath-not-below-mt'] = dat['mt-(glidepath-not-below-mt)']
            except:
                raise SyntaxError(
                    "neither mt-glidepath-not-below-mt or mt-(glidepath-not-below-mt) scenario name found in DWR_scenarios output")
        
        # Plot baseline scenario, only for first round
        if k == 0:
            dat['base'].plot(ax=ax[0],
                             label='Calibrated Run',
                             marker='o',
                             markevery=100,
                             markersize=5)
            dat.loc[(dat.index >= pd.to_datetime('01/01/2020')) &
                    (dat.index <= pd.to_datetime('01/01/2040')), 'base'].plot(ax=ax[1],
                                                                              label='Calibrated Run',
                                                                              marker='o',
                                                                              markevery=50,
                                                                              markersize=5)
        # Plot scenario data
        plot_keys = scenario_plot_keys[scen]
        # Full timeseries
        dat[plot_keys].plot(ax=ax[0])
        # Focus period
        dat.loc[(dat.index >= pd.to_datetime('01/01/2020')) &
                (dat.index <= pd.to_datetime('01/01/2040')), plot_keys].plot(ax=ax[1])

    # Plot observed subsidence
    sub_observed.reset_index().plot.scatter(ax=ax[0],
                                            x='Date',
                                            y='Subsidence_ft',
                                            c='k',
                                            s=6,
                                            label='Observed',
                                            zorder=10)
    # Format plot
    title = f'{site_name} - Scenario Results'
    for a in ax:
        a.set_ylim([a.get_ylim()[1], a.get_ylim()[0]])
        a.grid(True)
        a.set_xlabel('', fontsize=12)
    ax[0].set_ylabel('Subsidence (ft)', fontsize=12)
    ax[0].legend(ncols=2,
                 loc=3,
                 fontsize=10)
    ax[0].set_xlim(right=pd.to_datetime('01/01/2040'))
    ax[1].set_xlim([pd.to_datetime('01/01/2020'), pd.to_datetime('01/01/2040')])
    ax[1].set_ylabel('')
    ax[1].legend().remove()
    ax[1].set_xticks(ax[1].get_xticks())
    xticks = ax[1].get_xticklabels()
    xticks = [x if i % 2 == 0 else '' for i, x in enumerate(xticks)]
    ax[1].set_xticklabels(xticks, fontsize=10)
    ax[0].tick_params(axis='x', labelsize=10)
    ax[0].tick_params(axis='y', labelsize=10)
    ax[1].tick_params(axis='x', labelsize=10)
    ax[1].tick_params(axis='y', labelsize=10)
    
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.supxlabel('Year', y=-0.01, fontsize=12)
    
    savePath = os.path.join(m_d, 'figures')
    os.makedirs(savePath, exist_ok=True)
    
    plt.savefig(os.path.join(savePath, f"{site_name}_combined_scenarios_2040.png"),
                dpi=250,
                bbox_inches='tight')
    plt.close(fig)


def plot_WL_SUB_2_panel(m_d,
                        noptmax=None
                        ):
    """
    Parameters
    ----------
    m_d : str
        Model run directory name
    aquifer_keys : list, default is ['layer_3','Lower']
        List corresponding to layer and aquifer to be plotted. The first entry must
        match an available layer in the site's critical head data, and the second
        entry must match the corresponding aquifer name as it listed in the observed
        water lavel data.
    noptmax : int, default is None
        noptmax specifed for PEST run. If left as None, will automatically detect
        highest noptmax value from PEST file

    Returns
    -------
    None.

    """

    print("Creating WL/Sub/CH figure...")
    from pathlib import Path
    site_name = os.path.normpath(m_d).split(os.path.sep)[0]

    lith = pd.read_csv(os.path.join(site_name, "source_data", f"{site_name}_lithology.csv")).dropna(axis=1,
                                                                                                    how="all").dropna(
        axis=0)
    lith.Aquifer = lith.Aquifer.str.lower()
    lith_lays = [
        s.replace(" ", "").replace("aquifer", "").replace("plioscene", "pliocene").replace("composite", "corcoran")
        for s in lith.Aquifer.unique()]
    lith_nlay = len(lith.Aquifer.unique())
    layers = [f"layer_{l}" for l in range(1, len(lith_lays) + 1)]
    lith_dict = dict(zip(layers, lith_lays))
    print(f"{lith_nlay} layers in lithology file: {lith_dict}")

    # Build paths
    sub_file = os.path.join(site_name, 'source_data', f'{site_name}_sub_data.csv')
    ts_file = os.path.join(site_name, 'processed_data', f'{site_name}.ts_data.csv')
    # Check paths
    if not os.path.exists(m_d):
        raise FileNotFoundError(f"Specified model run path does not exist: {m_d}")

    # Get the highest valued mean_delayib_LowestGWL_{x}.csv available and check path
    # phidf = pd.read_csv(os.path.join(m_d, "pest.phi.actual.csv"))
    if noptmax is None:
    #     noptmax = phidf.iteration.max()
        noptmax = len([s for s in os.listdir(m_d) if s.startswith('mean_delayib_LowestGWL')])-1
    ch_path = os.path.join(m_d, f"mean_delayib_LowestGWL_{noptmax}.csv")

    # Load observed subsidence, WL, and estimated CH data
    sub_observed = pd.read_csv(sub_file,
                               index_col=0,
                               parse_dates=True)

    WL_observed = pd.read_csv(ts_file,
                              index_col=0,
                              parse_dates=True)
    WL_observed['Aquifer'] = WL_observed['Aquifer'].str.lower()

    CH_data = pd.read_csv(ch_path,
                          index_col=0,
                          parse_dates=True)

    for c in CH_data.keys():
        if c.startswith('layer'):
            if lith_dict[c] in WL_observed['Aquifer'].unique().tolist():
                # Initialize figure and plot results
                fig, ax = plt.subplots(2, 1,
                                       figsize=(12, 8))
                plt.subplots_adjust(hspace=0)
                ax2 = ax[0].twinx()

                # Plot critical head approximation
                CH_data[c].plot(ax=ax2,
                                color='midnightblue',
                                ls='--',
                                label=f'CH - {lith_dict[c].capitalize()} Aquifer ')
                WL_observed[WL_observed['Aquifer'] == lith_dict[c]]['interpolated'].plot(ax=ax2,
                                                                                         color='cornflowerblue',
                                                                                         ls='-',
                                                                                         label=f'GWL - {lith_dict[c].capitalize()}  Aquifer (ft)')

                # # Plot obs subsidence --> spirit leveling
                sub_observed['Subsidence_ft'] = sub_observed['Subsidence_ft'].astype(float)
                sub_observed = sub_observed.dropna()
                sub_observed['Subsidence_ft'].plot(ax=ax[0],
                                                   color='k',
                                                   marker='o',
                                                   label='Subsidence data (ft)'
                                                   )

                # # Merge CH with measured water levels and plot difference w/ fill_between
                merged = pd.merge(WL_observed,
                                  CH_data,
                                  left_index=True,
                                  right_index=True)
                merged = merged[merged['Aquifer'] == lith_dict[c]]
                merged['diff'] = merged['interpolated'] - merged[c]
                merged['diff'].plot(ax=ax[1],
                                    color='royalblue')
                # Fill between y=0 and the difference timeseries
                ax[1].fill_between(merged['diff'].index,
                                   merged['diff'],
                                   0,
                                   where=(merged['diff'] >= 0),
                                   color='green',
                                   alpha=0.5,
                                   )
                ax[1].fill_between(merged['diff'].index,
                                   merged['diff'],
                                   0,
                                   where=(merged['diff'] < 0),
                                   color='red',
                                   alpha=0.5,
                                   )

                # Figure formatting

                fig.suptitle(f'{site_name}', y=0.91, fontsize=13)

                h1, l1 = ax[0].get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax[0].legend(handles=h1[:2] + h2[:2],
                             labels=l1[:2] + l2[:2],
                             loc=3,
                             framealpha=0.8,
                             fontsize=10)

                ax[0].tick_params(direction='in', which='both', length=10)
                ax2.tick_params(direction='in', which='both', length=10)
                ax[0].tick_params(axis='x',
                                  which='both',
                                  bottom=False,
                                  labelbottom=False)
                ax[0].tick_params(axis='y',
                                  colors='k')

                ax[0].yaxis.label.set_color('k')
                ax2.tick_params(axis='y',
                                colors='royalblue')
                ax2.yaxis.label.set_color('royalblue')

                ax[0].set_ylabel('Land-surface\nsubsidence in feet', fontsize=12)
                ax[0].set_xlim([min(sub_observed.index[0], WL_observed.index[0]),
                                max(sub_observed.index[-1], WL_observed.index[-1])])
                ax2.set_ylabel('Water level altitude\nin feet above NAD88', fontsize=12)
                # Explicitly set label position
                ax2.yaxis.set_label_position("right")
                ax[0].set_ylim([ax[0].get_ylim()[1],
                                ax[0].get_ylim()[0]])

                ax[1].axhline(0, color='darkslategrey')
                ax[1].yaxis.label.set_color('k')
                ax[1].set_ylabel(f'GWL minus CH (ft)', fontsize=12)

                ax[1].tick_params(direction='in',
                                  which='both',
                                  length=10)
                ax[1].tick_params(labelbottom=True,
                                  labelleft=True)

                ax[0].tick_params(axis='y', labelsize=10)
                ax2.tick_params(axis='y', labelsize=10)
                ax[1].tick_params(axis='x', labelsize=10)
                ax[1].tick_params(axis='y', labelsize=10)

                savePath = os.path.join(m_d, "figures")
                os.makedirs(savePath, exist_ok=True)
                # plt.show()

                plt.savefig(os.path.join(savePath, f"{site_name}_WL_SUB_{lith_dict[c]}_{noptmax}.png"),
                            dpi=250,
                            bbox_inches='tight')

                plt.close(fig)


def export_realizations_to_dirs(t_d,m_d,real_name_tag="base",noptmax=None,just_mf6=True,
                                scenario_tag=""):
    assert os.path.exists(t_d)
    assert os.path.exists(m_d)
    location = t_d.split(os.path.sep)[0]
    print("...loading pst")
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    phidf = pd.read_csv(os.path.join(m_d,"pest.phi.actual.csv"))
    if noptmax is None:
        noptmax = phidf.iteration.max()
    else:
        assert noptmax in phidf.iteration.values
    pe_file = os.path.join(m_d,"pest.{0}.par.jcb".format(noptmax))
    print("...loading parameter ensemble ",pe_file)
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=pe_file)
    print(pe.index.tolist())
    use_reals = [r for r in pe.index if real_name_tag in r.lower()]
    assert len(use_reals) > 0
    print("...exporting realizations: ",str(use_reals))
    for use_real in use_reals:
        pst.parameter_data.loc[pe.columns,"parval1"] = pe.loc[use_real,:].values
        pst.control_data.noptmax = 0
        new_t_d = os.path.join(os.path.split(t_d)[0],location+"_"+\
                  os.path.split(t_d)[1]+"_"+use_real.replace(":","-"))+\
                  scenario_tag
        if os.path.exists(new_t_d):
            shutil.rmtree(new_t_d)
        shutil.copytree(t_d,new_t_d)
        pst.write(os.path.join(new_t_d,"pest.pst"),version=2)
        pyemu.os_utils.run("pestpp-ies pest.pst",cwd=new_t_d)
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_t_d)
        # sim_d = os.path.join(os.path.split(t_d)[0],"mf6_"+use_real.replace(":","-"))
        # if os.path.exists(sim_d):
        #     shutil.rmtree(sim_d)
        # sim_d.
        if just_mf6:
            sim.set_all_data_internal()
            shutil.rmtree(new_t_d)
        sim.write_simulation()
        # pre-run mf6
        bins = [b for b in os.listdir(bin_path) if b.startswith("mf6")]
        assert len(bins) == 1
        shutil.copy2(os.path.join(bin_path,bins[0]),os.path.join(new_t_d,bins[0]))
        pyemu.os_utils.run("mf6",cwd=new_t_d)
        
        # make sure we have only the pc mf6 version in the dir
        os.remove(os.path.join(new_t_d,bins[0]))
        shutil.copy2(os.path.join("bin","win","mf6.exe"),os.path.join(new_t_d,"mf6.exe"))

        #make a cheap plot, which also verifies that mf6 ran
        df = pd.read_csv(os.path.join(new_t_d,"model.csub.obs.csv"),index_col=0)        
        comp_cols = [c for c in df.columns if c.lower().startswith("compac")]
        print(comp_cols)
        df["total_compaction"] = df.loc[:,comp_cols].sum(axis=1)
        df.total_compaction.plot()
        plt.grid()
        ax = plt.gca()
        plt.tight_layout()
        ax.set_ylabel("total compaction")
        ax.set_xlabel("total simulation time")
        
        plt.savefig(os.path.join(new_t_d,"total_compaction.pdf"))
        plt.close("all")
        print("...exported realization ",use_real,"to dir ",new_t_d)



def plot_window_compare(results_d,tag=""):
    site_ds = [os.path.join(results_d,d) for d in os.listdir(results_d) if os.path.isdir(os.path.join(results_d,d))]
    print(site_ds)
    colors = ["m","c","b","g","gold"]
    for site_d in site_ds:
        master_ds = [os.path.join(site_d,d) for d in os.listdir(site_d) if
                     os.path.isdir(os.path.join(site_d,d)) and tag in d]
        master_ds.sort()
        print(master_ds)
        pt_results = []
        nzobs = []
        for i, master_d in enumerate(master_ds):

            pst = pyemu.Pst(os.path.join(master_d,"pest.pst"))
            obs = pst.observation_data
            sobs = obs.loc[obs.oname=="subdiff",:].copy()
            assert sobs.shape[0] > 0
            sobs["datetime1"] = pd.to_datetime(sobs.dt1)
            sobs["datetime2"] = pd.to_datetime(sobs.dt2)
            start_datetime = pd.to_datetime("1-feb-2024")
            end_datetime = start_datetime + pd.to_timedelta(365*20,unit='d')
            sobs = sobs.loc[sobs.datetime1>=start_datetime,:]
            sobs = sobs.loc[sobs.datetime2 <= end_datetime, :]

            sobs["dtdiff"] = (sobs.datetime2 - sobs.datetime1).dt.days
            msobs = sobs.loc[sobs.dtdiff==sobs.dtdiff.max(),:]
            assert len(msobs) == 1
            forecast = msobs.obsnme.iloc[0]
            # start_datetime = pd.to_datetime("1-feb-2024")
            # end_datetime = start_datetime + pd.to_timedelta(365*20,unit='d')
            #sobs = obs.loc[obs.usecol=="sim-subsidence-ft",:]
            nzobs.append(pst.nnz_obs)
            # assert sobs.shape[0] > 0
            # sobs["datetime"] = pd.to_datetime(sobs.datetime)
            # sobs["start_diff"] = np.abs((sobs.datetime-start_datetime).dt.days)
            # start_obs = sobs.loc[sobs.start_diff==sobs.start_diff.min(),"obsnme"].iloc[0]
            # sobs["end_diff"] = np.abs((sobs.datetime - end_datetime).dt.days)
            # end_obs = sobs.loc[sobs.end_diff == sobs.end_diff.min(), "obsnme"].iloc[0]

            kvobs = obs.loc[obs.usecol=="kv",:].copy()
            assert kvobs.shape[0] > 0
            kvobs["k"] = kvobs.k.astype(int)
            kvobs.sort_values(by="k",inplace=True)
            #keep = [forecast]
            #keep.extend(kvobs.obsnme.tolist())
            keep = kvobs.obsnme.tolist()
            keep.append(forecast)
            # keep.append(start_obs)
            # keep.append(end_obs)

            jcbs = [f for f in os.listdir(master_d) if f.endswith(".obs.jcb")]
            print(master_d,len(jcbs))
            assert len(jcbs) == 2
            jcbs.sort()
            pr = pyemu.ObservationEnsemble.from_binary(pst=None,filename=os.path.join(master_d,jcbs[0]))._df.loc[:,keep]
            pt = pyemu.ObservationEnsemble.from_binary(pst=None, filename=os.path.join(master_d, jcbs[1]))._df.loc[:,keep]
            pt_results.append(pt)

        fig,axes = plt.subplots(len(keep),1,figsize=(8.5,11))
        for kv,ax in zip(kvobs.obsnme,axes[:-1]):
            for master_d,pt,color,nz in zip(master_ds,pt_results,colors,nzobs):
                label = master_d.split("_")[-1].replace("window","")+"({0} obs)".format(nz)
                vals = np.log10(pt.loc[:,kv].values)
                ax.hist(vals,fc=color,alpha=0.5,label=label,density=True,bins=20)
                kde = stats.gaussian_kde(vals)
                xvals = np.linspace(vals.min(),vals.max(),200)
                kvals = kde(xvals)
                ax.plot(xvals,kvals,lw=2.0,color=color)
            ax.set_title(os.path.split(site_d)[1]+" "+kv,loc="left")
            ax.legend(loc="upper right")
            ax.set_yticks([])
            ax.set_xlabel("$log_{10}$ ib kv")

        ax = axes[-1]
        for master_d, pt, color, nz in zip(master_ds, pt_results, colors,nzobs):
            label = master_d.split("_")[-1].replace("window","")+"({0} obs)".format(nz)
            vals = pt.iloc[:, -1].values
            ax.hist(vals, fc=color, alpha=0.5, label=label,density=True,bins=20)
            kde = stats.gaussian_kde(vals)
            xvals = np.linspace(vals.min(),vals.max(),200)
            kvals = kde(xvals)
            ax.plot(xvals,kvals,lw=2.0,color=color)
        ax.set_title(os.path.split(site_d)[1]+" "+pr.columns.values[-1],loc="left")
        ax.set_yticks([])
        ax.legend(loc="upper right")
        ax.set_xlabel("delayed compaction")


        # ax = axes[-2]
        # for master_d, pt, color in zip(master_ds, pt_results, colors):
        #     label = master_d.split("_")[-1]
        #     ax.hist(np.log10(pt.iloc[:, -1].values), fc=color, alpha=0.5, label=label)
        # ax.set_title(os.path.split(site_d)[1]+" "+pr.columns.values[-2],loc="left")
        # ax.set_yticks([])
        # ax.legend(loc="upper right")
        # ax.set_xlabel("total compaction")
        #
        # ax = axes[-1]
        # for master_d, pt, color in zip(master_ds, pt_results, colors):
        #     label = master_d.split("_")[-1]
        #     ax.hist(np.log10(pt.iloc[:, -2].values), fc=color, alpha=0.5, label=label)
        # ax.set_title(os.path.split(site_d)[1] + " " + pr.columns.values[-1], loc="left")
        # ax.legend(loc="upper right")
        # ax.set_yticks([])
        # ax.set_xlabel("total compaction")
        plt.tight_layout()
        plt.savefig(os.path.join(site_d,os.path.split(site_d)[1]+tag+"_window_compare.pdf"))

        plt.close(fig)



def run_morris(org_t_d,new_t_d=None,num_workers=10,scenario_name="no-sgma",plusplus_kwargs={}):
    if new_t_d is None:
        new_t_d = org_t_d + "_morris"
        if scenario_name is not None:
            new_t_d += "_"+scenario_name
    if os.path.exists(new_t_d):
        shutil.rmtree(new_t_d)
    shutil.copytree(org_t_d,new_t_d)

    # we need to use an aggressive scenario 
    # so that the no-delay forecast actually does something. 
    # the no-sigma scenario looks like the one....
    pst = pyemu.Pst(os.path.join(new_t_d,"pest.pst"))
    par = pst.parameter_data
    if scenario_name is not None:
        pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(new_t_d,"scenarios.jcb"))
        print(pe.index)
        pe = pe.loc[pe.index.map(lambda x: scenario_name in x),:]
        assert pe.shape[0] > 0
        gpar = par.loc[par.pname=="directghbhead",:].copy()
        gpar["org_parval1"] = gpar.parval1.copy()
        assert gpar.shape[0] > 0
        par.loc[gpar.parnme,"parval1"] = pe.loc[pe.index[0],gpar.parnme].values
        #print(par.loc[gpar.parnme,"parval1"])
        #print(gpar.org_parval1)

 

    
    par.loc[par.pname.str.contains("ghb"),"partrans"] = "fixed"
    for key,val in plusplus_kwargs.items():
        pst.pestpp_options[key] = val
    pst.write(os.path.join(new_t_d,"pest.pst"),version=2)
    print(os.path.join(new_t_d,"pest.pst"))
    m_d = new_t_d.replace("template","master")
    pyemu.os_utils.start_workers(new_t_d,"pestpp-sen","pest.pst",
        num_workers=num_workers,master_dir=m_d,worker_root=".")
    return m_d


def plot_morris_delaydif_summary(m_d,npar_results=10):

    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    obs = pst.observation_data
    par = pst.parameter_data
    sdobs = obs.loc[obs.usecol.str.startswith("gwf-minus-delaylowest"),:].copy()
    sdobs["datetime"] = pd.to_datetime(sdobs.datetime)
    sdobs["k"] = sdobs.usecol.apply(lambda x: int(x.split(".")[1]))
    sddobs = sdobs.loc[sdobs.datetime.dt.year==2024,:].copy()
    phisen = pd.read_csv(os.path.join(m_d,"pest.msn"))
    df = pd.read_csv(os.path.join(m_d,"pest.mio"))
    uk = sdobs.k.unique()
    uk.sort()
    for k in uk:
        kobs = sddobs.loc[sdobs.k==k,:].copy()
        kobs.sort_values(by="datetime",inplace=True)
        forecast_name = kobs.obsnme.iloc[-1]

        
        phisen = pd.read_csv(os.path.join(m_d,"pest.msn"))
        df = pd.read_csv(os.path.join(m_d,"pest.mio"))
        foresen = df.loc[df.observation_name==forecast_name,:].copy()
        foresen.index = foresen.parameter_name
        phisen.index = phisen.parameter_name
        
        print(phisen["sen_mean_abs"])
        print(foresen["sen_mean_abs"])

        fig,axes = plt.subplots(2,1,figsize=(10,10))
        phisen.sort_values(by="sen_mean_abs",inplace=True,ascending=False)
        plot_phisen = phisen.iloc[:npar_results]
        foresen.sort_values(by="sen_mean_abs",inplace=True,ascending=False)
        plot_foresen = foresen.iloc[:npar_results]
        ax = axes[0]
        ax.scatter(plot_phisen.sen_mean_abs.values,plot_phisen.sen_std_dev.values,marker="o",
            s=10,c="b")
        for abs_sen,std_sen,pname in zip(plot_phisen.sen_mean_abs,plot_phisen.sen_std_dev,plot_phisen.index):
            ax.text(abs_sen,std_sen,get_common_par_name(pname,par,short=True),fontsize=6)
        ax.set_title("phi summary ",loc="left")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        mn = min(xlim[0],ylim[0])
        mx = max(xlim[1],ylim[1])
        lim = [mn,mx]
        ax.plot(lim,lim,"k--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_ylabel("sens standard deviation")
        ax.set_xlabel("sens mean")
        ax.grid()
        
        ax = axes[1]
        ax.scatter(plot_foresen.sen_mean_abs.values,plot_foresen.sen_std_dev.values,marker="o",
            s=10,c="b")
        for abs_sen,std_sen,pname in zip(plot_foresen.sen_mean_abs,plot_foresen.sen_std_dev,plot_foresen.index):
            ax.text(abs_sen,std_sen,get_common_par_name(pname,par,short=True),fontsize=6)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        mn = min(xlim[0],ylim[0])
        mx = max(xlim[1],ylim[1])
        lim = [mn,mx]
        ax.plot(lim,lim,"k--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        
        ax.set_ylabel("sens standard deviation")
        ax.set_xlabel("sens mean")

        ax.set_title("forecast summary\n"+forecast_name,loc="left")
        ax.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(m_d,"morris_1to1_{0}.pdf".format(forecast_name.replace(":","-"))))
 
        


        fig,ax = plt.subplots(1,1,figsize=(10,10))
        
        phisen["sen_mean_abs"] /= phisen["sen_mean_abs"].max()
        foresen["sen_mean_abs"] /= foresen["sen_mean_abs"].max()
        
        combosen = phisen.sen_mean_abs + foresen.sen_mean_abs
        combosen.sort_values(inplace=True,ascending=False)
        combosen = combosen.iloc[:npar_results]
      
        
        print(phisen.loc[combosen.index,"sen_mean_abs"])
        print(foresen.loc[combosen.index,"sen_mean_abs"])   
        #phisen.loc[combosen.index].plot(kind="bar",ax=ax)
        comb = pd.DataFrame(data={"phi":phisen.loc[combosen.index,"sen_mean_abs"],
                                  "forecast":foresen.loc[combosen.index,"sen_mean_abs"]})
        
        comb.index = comb.index.map(lambda x: get_common_par_name(x,par,short=True))
        comb.plot(kind="bar",ax=ax)
        label = "forecast:"+forecast_name
        ax.set_title(label,loc="left")
        ax.set_ylabel("scaled composite mean sensitivity")
        #ax.set_yticks([])
        ax.set_xlabel("")
        ax.grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(m_d,"morris_summary_{0}.pdf".format(forecast_name.replace(":","-"))))
        plt.close(fig)




def plot_morris(m_d,npar_results=10):
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    obs = pst.observation_data
    par = pst.parameter_data
    sdobs = obs.loc[obs.oname=="subdiff",:].copy()
    sdobs["dt1"] = pd.to_datetime(sdobs.dt1)
    sdobs["dt2"] = pd.to_datetime(sdobs.dt2)
    sddobs = sdobs.loc[sdobs.dt1.dt.year==2024,:].copy()
    sddobs.sort_values(by="dt2",inplace=True)
    forecast_name = sddobs.obsnme.iloc[-1]

    
    phisen = pd.read_csv(os.path.join(m_d,"pest.msn"))
    df = pd.read_csv(os.path.join(m_d,"pest.mio"))
    foresen = df.loc[df.observation_name==forecast_name,:].copy()
    foresen.index = foresen.parameter_name
    phisen.index = phisen.parameter_name
    


    print(phisen["sen_mean_abs"])
    print(foresen["sen_mean_abs"])
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    #axes = axes.flatten()
    #phisen["sen_mean_abs"] = np.log10(phisen.sen_mean_abs.values)
    #phisen["sen_std_dev"] = np.log10(phisen.sen_std_dev.values)
    phisen["sen_mean_abs"] /= phisen["sen_mean_abs"].max()

    #phisen.sort_values(by="sen_mean_abs",inplace=True,ascending=False)
    #phisen = phisen.iloc[:npar_results]
    
    #foresen["sen_mean_abs"] = np.log10(foresen.sen_mean_abs.values)
    #foresen["sen_std_dev"] = np.log10(foresen.sen_std_dev.values)
    #foresen.sort_values(by="sen_mean_abs",inplace=True,ascending=False)
    #foresen = foresen.iloc[:npar_results]
    foresen["sen_mean_abs"] /= foresen["sen_mean_abs"].max()
    
    combosen = phisen.sen_mean_abs + foresen.sen_mean_abs
    combosen.sort_values(inplace=True,ascending=False)
    combosen = combosen.iloc[:npar_results]
  
    # ax.scatter(phisen.sen_mean_abs,phisen.sen_std_dev,s=3)
    # for name,x,y in zip(phisen.parameter_name,
    #                     phisen.sen_mean_abs,
    #                     phisen.sen_std_dev):
    #     cname = get_common_par_name(name,par)
    #     ax.text(x,y,cname,bbox={"facecolor":'w',"alpha":1.0,"pad":0.1,"lw":0})
    # ax.grid()
    print(phisen.loc[combosen.index,"sen_mean_abs"])
    print(foresen.loc[combosen.index,"sen_mean_abs"])   
    #phisen.loc[combosen.index].plot(kind="bar",ax=ax)
    comb = pd.DataFrame(data={"phi":phisen.loc[combosen.index,"sen_mean_abs"],
                              "forecast":foresen.loc[combosen.index,"sen_mean_abs"]})
    
    comb.index = comb.index.map(lambda x: get_common_par_name(x,par,short=True))
    comb.plot(kind="bar",ax=ax)
    label = "Morris GSA summary: "+m_d
    ax.set_title(label,loc="left")
    ax.set_ylabel("scaled composite mean sensitivity")
    #ax.set_yticks([])
    ax.set_xlabel("")
    ax.grid()
    
    plt.tight_layout()
    plt.savefig(os.path.join(m_d,"morris_summary.pdf"))
    plt.close(fig)

    

def plot_morris_compare(m_ds):
    m_ds.sort()
    results = []
    for m_d in m_ds:
        pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
        obs = pst.observation_data
        sdobs = obs.loc[obs.oname=="subdiff",:].copy()
        sdobs["dt1"] = pd.to_datetime(sdobs.dt1)
        sdobs["dt2"] = pd.to_datetime(sdobs.dt2)
        sddobs = sdobs.loc[sdobs.dt1.dt.year==2024,:].copy()
        sddobs.sort_values(by="dt2",inplace=True)
        forecast_name = sddobs.obsnme.iloc[-1]

        
        phisen = pd.read_csv(os.path.join(m_d,"pest.msn"))
        df = pd.read_csv(os.path.join(m_d,"pest.mio"))
        foresen = df.loc[df.observation_name==forecast_name,:].copy()
        foresen.index = foresen.parameter_name
        phisen.index = phisen.parameter_name
        
        print(phisen["sen_mean_abs"])
        print(foresen["sen_mean_abs"])
        
        results.append([phisen,foresen,pst.parameter_data])
    npar_results = 10
    for m_d,result in zip(m_ds,results):
        if "nodelay" in m_d:
            continue
        
        phisen = result[0]
        foresen = result[1]
        par = result[2]
        fig,axes = plt.subplots(2,1,figsize=(5,5))
        #axes = axes.flatten()
        #phisen["sen_mean_abs"] = np.log10(phisen.sen_mean_abs.values)
        #phisen["sen_std_dev"] = np.log10(phisen.sen_std_dev.values)
        phisen["sen_mean_abs"] /= phisen["sen_mean_abs"].max()

        #phisen.sort_values(by="sen_mean_abs",inplace=True,ascending=False)
        #phisen = phisen.iloc[:npar_results]
        
        #foresen["sen_mean_abs"] = np.log10(foresen.sen_mean_abs.values)
        #foresen["sen_std_dev"] = np.log10(foresen.sen_std_dev.values)
        #foresen.sort_values(by="sen_mean_abs",inplace=True,ascending=False)
        #foresen = foresen.iloc[:npar_results]
        foresen["sen_mean_abs"] /= foresen["sen_mean_abs"].max()
        
        combosen = phisen.sen_mean_abs + foresen.sen_mean_abs
        combosen.sort_values(inplace=True,ascending=False)
        combosen = combosen.iloc[:npar_results]

        ax = axes[0]   
        # ax.scatter(phisen.sen_mean_abs,phisen.sen_std_dev,s=3)
        # for name,x,y in zip(phisen.parameter_name,
        #                     phisen.sen_mean_abs,
        #                     phisen.sen_std_dev):
        #     cname = get_common_par_name(name,par)
        #     ax.text(x,y,cname,bbox={"facecolor":'w',"alpha":1.0,"pad":0.1,"lw":0})
        # ax.grid()
        print(phisen.loc[combosen.index,"sen_mean_abs"])
        print(foresen.loc[combosen.index,"sen_mean_abs"])   
        #phisen.loc[combosen.index].plot(kind="bar",ax=ax)
        comb = pd.DataFrame(data={"phi":phisen.loc[combosen.index,"sen_mean_abs"],
                                  "forecast":foresen.loc[combosen.index,"sen_mean_abs"]})
        
        comb.index = comb.index.map(lambda x: get_common_par_name(x,par,short=True))
        comb.plot(kind="bar",ax=ax)
        label = "{0} delay formulation".format(os.path.split(m_d)[0])
        ax.set_title(label,loc="left")
    
        
        m_d = m_d.replace("delay","nodelay")
        result = results[m_ds.index(m_d)]
        phisen = result[0]
        foresen = result[1]
        par = result[2]

        #phisen["sen_mean_abs"] = np.log10(phisen.sen_mean_abs.values)
        #phisen["sen_std_dev"] = np.log10(phisen.sen_std_dev.values)
        phisen["sen_mean_abs"] /= phisen["sen_mean_abs"].max()

        #phisen.sort_values(by="sen_mean_abs",inplace=True,ascending=False)
        #phisen = phisen.iloc[:npar_results]
        
        #foresen["sen_mean_abs"] = np.log10(foresen.sen_mean_abs.values)
        #foresen["sen_std_dev"] = np.log10(foresen.sen_std_dev.values)
        #foresen.sort_values(by="sen_mean_abs",inplace=True,ascending=False)
        #foresen = foresen.iloc[:npar_results]
        foresen["sen_mean_abs"] /= foresen["sen_mean_abs"].max()
        
        combosen = phisen.sen_mean_abs + foresen.sen_mean_abs
        combosen.sort_values(inplace=True,ascending=False)
        combosen = combosen.iloc[:npar_results]

        ax = axes[1]   
        # ax.scatter(phisen.sen_mean_abs,phisen.sen_std_dev,s=3)
        # for name,x,y in zip(phisen.parameter_name,
        #                     phisen.sen_mean_abs,
        #                     phisen.sen_std_dev):
        #     cname = get_common_par_name(name,par)
        #     ax.text(x,y,cname,bbox={"facecolor":'w',"alpha":1.0,"pad":0.1,"lw":0})
        # ax.grid()
        print(phisen.loc[combosen.index,"sen_mean_abs"])
        print(foresen.loc[combosen.index,"sen_mean_abs"])   
        #phisen.loc[combosen.index].plot(kind="bar",ax=ax)
        comb = pd.DataFrame(data={"phi":phisen.loc[combosen.index,"sen_mean_abs"],
                                  "forecast":foresen.loc[combosen.index,"sen_mean_abs"]})
        comb.index = comb.index.map(lambda x: get_common_par_name(x,par,short=True))
        comb.plot(kind="bar",ax=ax)
        label = "{0} no-delay formulation".format(os.path.split(m_d)[0])
        ax.set_title(label,loc="left")
        for ax in axes:
            ax.set_ylabel("scaled composite mean sensitivity")
            #ax.set_yticks([])
            ax.set_xlabel("")
    
        plt.tight_layout()
        plt.savefig(os.path.join(m_d,"morris_compare.pdf"))
        plt.close(fig)


def get_common_par_name(name,par,short=False):
    cname = ''
    if not short and par.loc[name,"partrans"] == "log":
        cname += "$log_{10}$ "
    

    if "pname:csub" in name:
        layer = int(par.loc[name,"k"]) + 2
        cname += name.split("usecol:")[1].split("pstyle")[0].replace("_"," ").strip() + " layer {0}".format(layer)
    elif "pname:cg" in name:
        layer = int(name.split("lyr")[1].split('_')[0])
        cname += name.split("_")[1] + " layer {0}".format(layer)
    elif "pname:k33" in name:
        layer = int(par.loc[name,"k"]) + 1
        cname += "k33 layer {0}".format(layer)
    elif "consthead" in name:
        cname += "gw level"
    else:
        cname += " " + name
    #cname = cname.replace("clay","clay thickness")
    cname = cname.replace("kv","interbed vk")
    cname = cname.replace("k33","aquifer vk")

    if not short and par.loc[name,"pstyle"] == "m":
        cname += " multiplier"
    return cname

def make_combined_posterior(m_d):
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"),parse_metadata=False)
    phidf = pd.read_csv(os.path.join(m_d,"pest.phi.actual.csv"),index_col=0)
    pes,oes = [],[]
    prior_nreal = None
    for itr in phidf.index.values:
        if itr == "combined":
            continue
        pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.{0}.par.jcb".format(itr)))
        oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.{0}.obs.jcb".format(itr)))
        if itr == 0:
            pr_pv = oe.phi_vector
            prior_nreal = pe.shape[0]
            
        pe = pe._df
        oe = oe._df
        pe.index = pe.index.map(lambda x: "real{0}_itr{1}".format(x,itr))
        oe.index = oe.index.map(lambda x: "real{0}_itr{1}".format(x,itr))        
        pes.append(pe)
        oes.append(oe)

    if len(pes) == 1:
        pyemu.ParameterEnsemble(df=pes[0],pst=pst).to_binary(os.path.join(m_d,"pest.combined.par.jcb"))
        pyemu.ObservationEnsemble(df=oes[0],pst=pst).to_binary(os.path.join(m_d,"pest.combined.obs.jcb"))
        return
    

    oe = pd.concat(oes)
    pe = pd.concat(pes)

    oe = pyemu.ObservationEnsemble(df=oe,pst=pst)
    pv = oe.phi_vector
    pv.sort_values(inplace=True)
    pv = pv.iloc[:prior_nreal]

    base_idxs = oe.loc[oe.index.map(lambda x: "realbase" in x),:].index.values
    base_pv = oe.loc[base_idxs,:].phi_vector
    min_base_real = None
    if base_pv.shape[0] > 0:
        min_base_real = base_pv.loc[base_pv==base_pv.min()].index[0]
    #print(min_base_real)
    base_pe = pe.loc[base_idxs].copy()
    base_oe = oe.loc[base_idxs].copy()
    
    #print(oe.shape)
    #print(pe.shape)
    #print(pv) 
    pe = pe.loc[pv.index,:]
    oe = oe.loc[pv.index,:]
    if min_base_real is not None:
        pe.loc["base",:] = base_pe.loc[min_base_real,:].values
        oe.loc["base",:] = base_oe.loc[min_base_real,:].values
        assert "base" in pe.index
        assert "base" in oe.index
    else:
        print("WARNING: no base realizations found...")

    pyemu.ParameterEnsemble(df=pe,pst=pst).to_binary(os.path.join(m_d,"pest.combined.par.jcb"))
    #print(oe)
    #oe = pyemu.ObservationEnsemble(df=oe,pst=pst)
    oe.to_binary(os.path.join(m_d,"pest.combined.obs.jcb"))
    with PdfPages(os.path.join(m_d,"phi_combined.pdf")) as pdf:
        fig,ax = plt.subplots(1,1,figsize=(6,3))
        colors = ["0.5","b"]
        
        # icolor = 0
        ax.hist(np.log10(pr_pv.values),bins=20,alpha=0.5,facecolor="0.5")
        ax.hist(np.log10(oe.phi_vector.values),bins=20,alpha=0.5,facecolor="b")
        ax.set_title("prior and combined-posterior phi histograms",loc="left")
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)


def plot_compare_model_input_summary(m_ds,noptmax=None):
    results = []

    for m_d in m_ds:
        pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
        phidf = pd.read_csv(os.path.join(m_d,"pest.phi.actual.csv"))
        if noptmax is None:
            noptmax = phidf.iteration.max()
        pr = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.0.obs.jcb"))
        pt = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.{0}.obs.jcb".format(noptmax)))
        results.append([pst,pr,pt,m_d])

    common_names = []
    for result in results:
        obs = result[0].observation_data
        oobs = obs.loc[obs.oname.apply(lambda x: x in ["listobs","arrobs"]),"obsnme"].tolist()
        common_names.extend(oobs)
        common_names = list(set(common_names))
        #print(oobs)

        
    common_names.sort()
    print(common_names)
    
    with PdfPages(os.path.join(m_d,"input_summary_compare_{0}.pdf".format(noptmax))) as pdf:
        
        for oname in common_names:
            if "lower_bound" in oname:
                continue
            if "upper_bound" in oname:
                continue
            if "usecol:org" in oname:
                continue
            if "_squared" in oname:
                continue
            print("...",oname)
            fig,ax = plt.subplots(1,1,figsize=(7,3))
            colors = ["m","g"]
            for (pst,pr,pt,m_d),color in zip(results,colors):
                if oname not in pst.obs_names:
                    continue

                ax.set_title(oname,loc="left")

                
                en = pt
                label = "delay formulation"
                if "nodelay" in m_d:
                    label = "no-delay formulation"
                if min(en.loc[:,oname].values.min(),en.loc[:,oname].values.min()) <=0.0:
                    
                    ax.hist(en.loc[:,oname].values,bins=20,alpha=0.5,facecolor=color,density=True,label=label)
                   
                else:
                    ax.hist(np.log10(en.loc[:,oname].values),bins=20,alpha=0.5,facecolor=color,density=True,label=label)
                   
                 

                #ax.set_yticks([])
            ax.legend(loc="upper right")
            ax.grid()
                #ax.set_yticklabels(["" for _ in ax.get_yticks()])
        
                
                    
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

def plot_compare_parameter_summary(m_ds,noptmax=None):
    results = []

    for m_d in m_ds:
        pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
        phidf = pd.read_csv(os.path.join(m_d,"pest.phi.actual.csv"))
        if noptmax is None:
            noptmax = phidf.iteration.max()
        pr = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.0.par.jcb"))
        pt = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.{0}.par.jcb".format(noptmax)))
        results.append([pst,pr,pt,m_d])

    common_names = []
    for result in results:
        par = result[0].parameter_data
        ppar = par.loc[par.pname.apply(lambda x: x in ["csub","cg","k33"]),"parnme"].tolist()
        common_names.extend(ppar)
        common_names = list(set(common_names))
        #print(oobs)

        
    common_names.sort()
    print(common_names)
    
    with PdfPages(os.path.join(m_d,"parameter_compare_{0}.pdf".format(noptmax))) as pdf:
        
        for oname in common_names:
            if pst.parameter_data.loc[oname,"partrans"] == "fixed":
                    continue 
            # if "lower_bound" in oname:
            #     continue
            # if "upper_bound" in oname:
            #     continue
            # if "usecol:org" in oname:
            #     continue
            # if "_squared" in oname:
            #     continue
            print("...",oname)
            fig,ax = plt.subplots(1,1,figsize=(7,3))
            colors = ["m","g"]
            plot_prior = True
            for (pst,pr,pt,m_d),color in zip(results,colors):
                if oname not in pst.par_names:
                    continue

                ax.set_title(get_common_par_name(oname,pst.parameter_data),loc="left")

                
                en = pt
                label = "delay formulation"
                if "nodelay" in m_d:
                    label = "no-delay formulation"
                #if min(en.loc[:,oname].values.min(),en.loc[:,oname].values.min()) <=0.0:
                
                if pst.parameter_data.loc[oname,"partrans"] != "log":    
                    ax.hist(en.loc[:,oname].values,bins=20,alpha=0.5,facecolor=color,density=True,label=label)
                   
                else:
                    ax.hist(np.log10(en.loc[:,oname].values),bins=20,alpha=0.5,facecolor=color,density=True,label=label)
                   
                if plot_prior and "pcs0" not in oname:
                    en = pr
                    #if min(en.loc[:,oname].values.min(),en.loc[:,oname].values.min()) <=0.0:
                    if pst.parameter_data.loc[oname,"partrans"] != "log":    
                        ax.hist(en.loc[:,oname].values,bins=20,alpha=0.5,facecolor="0.5",density=True,label="prior")
                   
                    else:
                        ax.hist(np.log10(en.loc[:,oname].values),bins=20,alpha=0.5,facecolor="0.5",density=True,label="prior")
                    plot_prior = False
                


                #ax.set_yticks([])
            ax.semilogy()
            ax.set_ylabel("count")
            ax.legend(loc="upper right")
            ax.grid()
                #ax.set_yticklabels(["" for _ in ax.get_yticks()])
        
                
                    
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)


def plot_compare_sub(m_ds, noptmax=None):
    
    
    # test = pd.read_csv(os.path.join(site_name, 'model_ws', 'baseline', 'J88.baseline.csub.obs.csv'))
    # add_cols = [c for c in test.columns if 'COMPACTION' in c]
    # test.loc[:, 'sim_subsidence_ft'] = test.loc[:, add_cols].sum(axis=1)
    results = []
    for m_d in m_ds:
        site = os.path.split(m_d)[0]
        output_file = os.path.join("Output","{0}_AllLayers.csv".format(site.upper()))
        assert os.path.exists(output_file)
        out_df = pd.read_csv(output_file)
        out_df.index = pd.to_datetime(out_df.pop("Date"))
        out_df.index.name = "datetime"
        out_df["subsidence_ft"] = out_df.pop("Sub")

        pst = pyemu.Pst(os.path.join(m_d, "pest.pst")) 
        phidf = pd.read_csv(os.path.join(m_d,"pest.phi.actual.csv"))
        if noptmax is None:
            noptmax = phidf.iteration.max()
        pr = pyemu.ObservationEnsemble.from_binary(pst,
                                                      os.path.join(m_d,
                                                                   'pest.0.obs.jcb'))

        
        pt = pyemu.ObservationEnsemble.from_binary(pst,
                                                      os.path.join(m_d,
                                                                   f'pest.{noptmax}.obs.jcb'))
        
        assert pt.shape[1] == pst.nobs
        noise = pyemu.ObservationEnsemble.from_binary(pst,
                                                        os.path.join(m_d,
                                                                    'pest.obs+noise.jcb'))
     
        obs = pst.observation_data
        sobs = obs.loc[obs.usecol=="sim-subsidence-ft",:].copy()
        assert sobs.shape[0] > 0
        sobs["datetime"] = pd.to_datetime(sobs.datetime)
        obsobs = sobs.loc[sobs.observed==True,:].copy()
        nzobs = sobs.loc[sobs.weight > 0, :].copy()

        nzobs.sort_values(by="datetime",inplace=True)
        obsobs.sort_values(by="datetime", inplace=True)

        #find the nearest observed sub to the out_df datetime 
        obsobs["distance"] = np.abs((obsobs.datetime - out_df.index.values[0]).dt.days)
        offset = obsobs.loc[obsobs.distance==obsobs.distance.min(),"obsval"].values
        
        #print(offset)
        out_df["subsidence_ft"] *= 3.281
        out_df["subsidence_ft"] += offset
        #print(out_df.subsidence_ft)
        
        #print(obsobs.distance)


        results.append([pst,pr,pt,noise,sobs,obsobs,nzobs,out_df])
        



    with PdfPages(os.path.join(m_d, f"compare_compaction_summary_{noptmax}.pdf")) as pdf:  
        fig, axes = plt.subplots(2, 1, figsize=(6,6))
        ymax = -1e30   
        for m_d,ax,(pst,pr_en,pt_en,noise_en,sobs,obsobs,nzobs,out_df) in zip(m_ds,axes,results):
            site = os.path.split(m_d)[0]
            label = site + " delay formulation"
            if "nodelay" in m_d:
                label = site + " no-delay formulation"
            ax.set_title(label,loc="left")
            # plot non-zero weighted observations by measurement type
            color = cm.jet(np.linspace(0, 1, len(obsobs.source.unique())))
            dts = nzobs.datetime.values
            vals = noise_en.loc[:,nzobs.obsnme].values
            [ax.plot(dts, vals[i,:], color='r', lw=0.025, alpha=0.1,label="noise") for i in range(vals.shape[0])]
            

            dts = sobs.datetime.values
            vals = pr_en.loc[:,sobs.obsnme].values
            [ax.plot(dts, vals[i,:], color='0.5', lw=0.025, alpha=0.1,
                     label="prior") for i in range(vals.shape[0])]
            vals = pt_en.loc[:,sobs.obsnme].values
            [ax.plot(dts, vals[i, :], color='b', lw=0.025, alpha=0.1,
                     label='posterior') for i in range(vals.shape[0])]
            ymax = max(ymax,obsobs.obsval.max(),vals.max())

            for i,mtype in enumerate(obsobs.source.unique()):
                tmp = nzobs.loc[nzobs.source==mtype,:].copy()
                tmp = tmp.loc[tmp.weight > 0, :]
                #if tmp.shape[0] > 0:
                tmp.sort_values(by="datetime", inplace=True)
                ax.scatter(tmp.datetime.values, tmp.obsval.values, marker="o", c='r', s=4,
                        label="observed",zorder=10)
            print(out_df.index)
            print(out_df.subsidence_ft)
            ax.plot(out_df.index,out_df.subsidence_ft,"m-",label="3D model",alpha=0.5,zorder=1)
            
            
        xlim = axes[0].get_xlim()

        for iax,ax in enumerate(axes):
            handles, labels = ax.get_legend_handles_labels()
            handelz, labelz = [], []
            for i,l in enumerate(labels):
                if l not in labelz:
                    labelz.append(l)
                    handelz.append(handles[i])
            ax.legend(handles=handelz, labels=labelz,loc="upper left")    
            ax.set_xlim(xlim)
            ax.set_ylim(0,ymax*1.1)
            ax.set_ylabel("land subsidence (feet)")
            ax.grid()
            
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)

        
def xfer_from_m_d(tpl_dir,xfer_m_d,xfer_noptmax=None):
    xpst = pyemu.Pst(os.path.join(xfer_m_d,"pest.pst"))
    if xfer_noptmax is None:
        #make_combined_posterior(xfer_m_d)
        #try:
        #    _plot_thread(master_dir=xfer_m_d, i="combined")
        #except:
        #    pass
        #xpe = pyemu.ParameterEnsemble.from_binary(pst=xpst,filename=os.path.join(xfer_m_d,"pest.combined.par.jcb"))._df
        phidf = pd.read_csv(os.path.join(xfer_m_d,"pest.phi.actual.csv"))
        xfer_noptmax = phidf.iteration.max()

    #else:
    xpe = pyemu.ParameterEnsemble.from_binary(pst=xpst, filename=os.path.join(xfer_m_d,
                                              "pest.{0}.par.jcb".format(xfer_noptmax)))._df
    pst = pyemu.Pst(os.path.join(tpl_dir,"pest.pst"),parse_metadata=False)
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(tpl_dir,"prior_pe.jcb"))._df
    par = pst.parameter_data
    
    if pe.shape[0] > xpe.shape[0]:
        pe = pe.iloc[:xpe.shape[0],:]   
    elif pe.shape[0] < xpe.shape[0]:
        drop_real = [idx for idx in xpe.index if idx != "base"][-1]
        xpe.drop(drop_real,inplace=True)
    if "base" not in pe.index and "base" in xpe.index:
        pe = pe.iloc[:-1,:]
        pe.loc["base",:] = pst.parameter_data.parval1.loc[pe.columns]
    

    xpar = xpst.parameter_data
    par = pst.parameter_data
    xproppar = xpar.loc[~xpar.parnme.str.contains("ghb"),:]
    #xproppar = xproppar.loc[~xproppar.parnme.str.contains("consthead"),:]
    proppar = par.loc[~par.parnme.str.contains("ghb"),:]
    #proppar = proppar.loc[~proppar.parnme.str.contains("consthead"),:]
    
    assert xproppar.shape[0] > 0
    assert proppar.shape[0] > 0
    missing = set(proppar.parnme.tolist()).symmetric_difference(set(xproppar.parnme.tolist()))
    print(missing)
    assert len(missing) == 0
    #assert xproppar.shape[0] == proppar.shape[0]    
    #print(pe.shape,pst.npar,pst.npar_adj)

    pe.index = xpe.index.values
    #print(pe.index)

    pe.loc[:,xproppar.parnme.values] = xpe.loc[:,xproppar.parnme.values].values
    pe = pyemu.ParameterEnsemble(df=pe,pst=None)
    #pe = pe.loc[:,pst.adj_par_names]
    pe.to_binary(os.path.join(tpl_dir,"xfer.jcb"))
    #print(pe.iloc[0,:].values)
    pst.pestpp_options["ies_par_en"] = "xfer.jcb"
    #pst.control_data.noptmax = -1

    noise = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(tpl_dir,"noise.jcb"))
    if noise.shape[0] > pe.shape[0]:
        noise = noise.iloc[:pe.shape[0],:]
        noise._df.index = pe.index.values
    noise.to_binary(os.path.join(tpl_dir,"noise_xfer.jcb"))
    pst.pestpp_options["ies_obs_en"] = "noise_xfer.jcb"
    
    pst.pestpp_options["debug_parse_only"] = True
    pst.write(os.path.join(tpl_dir,"pest.pst"),version=2)
    pyemu.os_utils.run("pestpp-ies pest.pst",cwd=tpl_dir)
    pst.pestpp_options.pop("debug_parse_only")
    pst.write(os.path.join(tpl_dir,"pest.pst"),version=2)



def _plot_compaction(axes,pst,itr,label,color):

    oe = pst.ies.obsen
    pr_en = oe.loc[oe.index.get_level_values(0)==0,:]
    pt_en = oe.loc[oe.index.get_level_values(0)==itr,:]
    
    print(pt_en.shape)
    assert pt_en.shape[1] == pst.nobs
    noise_en = pst.ies.noise
 
    obs = pst.observation_data
    sobs = obs.loc[obs.usecol=="sim-subsidence-ft",:].copy()
    assert sobs.shape[0] > 0
    sobs["datetime"] = pd.to_datetime(sobs.datetime)
    cobs = obs.loc[obs.usecol.str.startswith("compaction."),:].copy()
    assert cobs.shape[0] > 0
    sobs["datetime"] = pd.to_datetime(sobs.datetime,format="%Y%m%d")
    cobs["datetime"] = pd.to_datetime(cobs.datetime,format="%Y%m%d")
    
    
    pobs = obs.loc[obs.usecol.str.contains("preconstress"),:].copy()
    assert pobs.shape[0] > 0

    pobs["datetime"] = pd.to_datetime(pobs.datetime,format="%Y%m%d")

    hobs = obs.loc[obs.oname=="hds",:].copy()
    assert hobs.shape[0] > 0
    hobs["k"] = hobs.usecol.apply(lambda x: int(x.split(".")[1])) - 1
    hobs["datetime"] = pd.to_datetime(hobs.datetime)

    dobs = obs.loc[obs.usecol.str.startswith("delay-head"),:]
    assert dobs.shape[0] > 0
    dobs["k"] = dobs.usecol.apply(lambda x: int(x.split(".")[1])) - 1
    dobs["datetime"] = pd.to_datetime(dobs.datetime)
    uks = dobs.k.unique()

    obsobs = sobs.loc[sobs.observed==True,:].copy()
    nzobs = sobs.loc[sobs.weight > 0, :].copy()

    nzobs.sort_values(by="datetime",inplace=True)
    obsobs.sort_values(by="datetime", inplace=True)
 
    ax = axes[0]
    dts = nzobs.datetime.values
    vals = noise_en.loc[:,nzobs.obsnme].values
    [ax.plot(dts, vals[i,:], color='r', lw=0.05, alpha=0.1,label="noise") for i in range(vals.shape[0])]
    

    dts = sobs.datetime.values
    #vals = pr_en.loc[:,sobs.obsnme].values
    #[ax.plot(dts, vals[i,:], color=color, ls="--",lw=0.025, alpha=0.1,
    #         label="prior reals") for i in range(vals.shape[0])]
    vals = pt_en.loc[:,sobs.obsnme].values
    [ax.plot(dts, vals[i, :], color=color, lw=0.5, alpha=0.5,
             label='posterior ' + label) for i in range(vals.shape[0])]

    if "base" in pt_en.index.get_level_values(1).tolist():
        axes[0].plot(dts, pt_en.loc[pt_en.index.get_level_values(1)=="base", sobs.obsnme].values.flatten(), color=color, lw=2.5,
                label='base posterior '+label)

    #if "base" in pr_en.index:
    #    axes[0].plot(dts, pr_en.loc["base", sobs.obsnme], color=color, ls="--", lw=2.5,
    #                        label='base prior')

    for i,mtype in enumerate(obsobs.source.unique()):
        tmp = nzobs.loc[nzobs.source==mtype,:].copy()
        tmp = tmp.loc[tmp.weight > 0, :]
        #if tmp.shape[0] > 0:
        tmp.sort_values(by="datetime", inplace=True)
        axes[0].scatter(tmp.datetime.values, tmp.obsval.values, marker="o", c='r', s=4,
                label="assimilated",zorder=10)
        

        tmp2 = obsobs.loc[obsobs.source == mtype, :].copy()
        tmp2.sort_values(by="datetime", inplace=True)
        diff = list(set(tmp.obsnme.tolist()).symmetric_difference(tmp2.obsnme.tolist()))

        tmpd = obsobs.loc[diff,:].copy()
        tmpd.sort_values(by="datetime",inplace=True)

        axes[0].scatter(tmpd.datetime.values, tmpd.obsval.values, marker="o", facecolors="none",alpha=0.5,edgecolors='r', s=50,
                        zorder=10,label="observed")
        
    axes[0].set_title(' total compaction obs vs sim with noise', loc="left")

    
    xlim = axes[0].get_xlim()
    
    uks = dobs.k.unique()
    uks.sort()
    
    for ik,uk in enumerate(uks):
        uhobs = hobs.loc[hobs.k==uk,:].copy()
        uhobs.sort_values(by="datetime",inplace=True)
        udobs = dobs.loc[dobs.k==uk,:].copy()
        udobs.sort_values(by="datetime",inplace=True)

        hdts = uhobs.datetime.values
        pr_hvals = pr_en.loc[:,uhobs.obsnme].values
        hvals = pt_en.loc[:,uhobs.obsnme].values
        #[ax.plot(hdts,hvals[i,:],"0.5",lw=0.1,alpha=0.3) for i in range(hvals.shape)]
        ddts = udobs.datetime.values
        dvals = pr_en.loc[:,udobs.obsnme].values
        #dvals[np.abs(dvals)>1e30] = np.nan
        dvals[dvals==-999] = np.nan
        prlowest = []
        for i in range(dvals.shape[0]):
            lowest = []
            ddvals = dvals[i,:]
            for j in range(1,ddvals.shape[0]):
                lowest.append(np.nanmin(ddvals[:j]))
            prlowest.append(np.array(lowest))
        prlowest = np.array(prlowest)
        
        dvals = pt_en.loc[:,udobs.obsnme].values
        dvals[dvals==-999] = np.nan
        ptlowest = []
        for i in range(dvals.shape[0]):
            lowest = []
            ddvals = dvals[i,:]
            for j in range(1,ddvals.shape[0]):
                lowest.append(np.nanmin(ddvals[:j]))
            ptlowest.append(np.array(lowest))
        ptlowest = np.array(ptlowest)


        ax = axes[ik+1]
        prlowest_diff = pr_hvals[:,1:] - prlowest
        ptlowest_diff = hvals[:,1:] - ptlowest
        #[ax.plot(ddts[1:],prlowest_diff[i,:],color,ls="--",lw=0.1,alpha=0.3) for i in range(prlowest_diff.shape[0])]
        [ax.plot(ddts[1:],ptlowest_diff[i,:],color,lw=0.1,alpha=0.3,label=label) for i in range(ptlowest_diff.shape[0])]
        if "base" in pt_en.index.get_level_values(1).tolist(): 
            
            bidx = pt_en.index.get_level_values(1).tolist().index("base")
            ax.plot(ddts[1:], ptlowest_diff[bidx,:], color=color, lw=2.5,
                    label='base posterior '+label)
        ax.set_title("gwf gw level minus delay ib lowest gw level layer {0}".format(uk+1),loc="left")
        #ax.grid()
    for iax,ax in enumerate(axes):
        handles, labels = ax.get_legend_handles_labels()
        handelz, labelz = [], []
        for i,l in enumerate(labels):
            if l not in labelz:
                labelz.append(l)
                handelz.append(handles[i])
        ax.legend(handles=handelz, labels=labelz)    
        ax.set_xlim(xlim)
        ax.grid()
        

   
    #return fig,axes
    
    

def _plot_compare(itr,uks,arr_onames,list_onames,figure_tag,m_ds,labels,colors):

    psts = []
    #print(pyemu.__file__)
    for m_d in m_ds:
        pst = pyemu.Pst(os.path.join(m_d,"pest.pst"),result_dir=m_d)
        psts.append(pst)
        #print(m_d,psts[-1].ies)
    
    with PdfPages(os.path.join(figure_tag+"_input_summary_{0}_compare.pdf".format(itr))) as pdf:
        figs,axes_list = [],[]
        fig, axes = plt.subplots(len(uks)+1,1,figsize=(8.5,11))
        for m_d,pst,label,color in zip(m_ds,psts,labels,colors):
            _plot_compaction(axes,pst,itr,label,color)
           
        for ax in axes:
            ax.grid()
        plt.tight_layout()
        pdf.savefig(fig)

        plt.close(fig)

        for oname in list_onames:
            print("...",oname)
            #first plot all together - pr and itr
            
            fig,axes = plt.subplots(2,1,figsize=(7,7))
            axt0 = axes[0].twinx()
            axt1 = axes[1].twinx()
            for m_d,pst,label,color in zip(m_ds,psts,labels,colors):
                obs = pst.observation_data
                if oname not in obs.obsnme.values:
                    print("oname {0} missing from {1}".format(oname,m_d))
                    continue
                oe = pst.ies.obsen
                #print(oe.index.get_level_values(1))
                pr = oe.loc[oe.index.get_level_values(0)==0,:]
                pt = oe.loc[oe.index.get_level_values(0)==itr,:]
                #print(pr.shape,pt.shape)
                if pr.shape[0] == 0:
                    raise Exception("prior missing")
                if pt.shape[0] == 0:
                    continue
                #print(m_d)
                xvals = np.linspace(min(pr.loc[:,oname].values.min(),pt.loc[:,oname].values.min()),
                                    max(pr.loc[:,oname].values.max(),pt.loc[:,oname].values.max()),200)
                try:

                    kde = stats.gaussian_kde(pt.loc[:,oname].values)
                    yvals = kde(xvals)
                except:
                    yvals = np.array([np.nan for _ in xvals])
                
                
                if min(pr.loc[:,oname].values.min(),pr.loc[:,oname].values.min()) <=0.0:
                    
                    axt1.hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="none",hatch="X", edgecolor=color,density=True)
                    axes[1].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor=color,density=True)
                    axes[1].plot(xvals,yvals,color=color,lw=2.0,label=label)
                    axes[1].set_title(oname,loc="left")

                else:
                    axt1.hist(np.log10(pr.loc[:,oname].values),bins=20,alpha=0.5,facecolor="none",hatch="X", edgecolor=color,density=True)
                    axes[1].hist(np.log10(pt.loc[:,oname].values),bins=20,alpha=0.5,facecolor=color,density=True)
                    xvalslog = np.linspace(np.log10(xvals.min()),np.log10(xvals.max()),200)
                    try:
                        kde = stats.gaussian_kde(np.log10(pt.loc[:,oname].values))
                        yvalslog = kde(xvalslog)
                    except:
                        yvalslog = np.array([np.nan for _ in xvals])
                    axes[1].plot(xvalslog,yvalslog,color=color,lw=2.0,label=label)
                    axes[1].set_title(oname+" (log10)",loc="left")
                    ylim = axes[1].get_ylim()
                
                axt0.hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="none",hatch="X", edgecolor=color,density=True)
                axes[0].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor=color,density=True)
                axes[0].plot(xvals,yvals,color=color,lw=2.0,label=label)

                axes[0].set_title(oname,loc="left")
                axes[0].legend(loc="upper right")
                axes[1].legend(loc="upper right")
                
               
                #ylim = axes[0].get_ylim()
                #axes[0].plot([lower,lower],ylim,"k--",lw=2)
                #axes[0].plot([upper,upper],ylim,"k--",lw=2)
                axes[0].set_yticks([])
                axes[1].set_yticks([])
                axt0.set_yticks([])
                axt1.set_yticks([])
                ylim = axt0.get_ylim()
                axt0.set_ylim(ylim[0],ylim[1]*1.5)
                ylim = axt1.get_ylim()
                axt1.set_ylim(ylim[0],ylim[1]*1.5)
                
            
            
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
            
       
        for oname in arr_onames:
            print("...",oname)
            #first plot all together - pr and itr
            
            fig,axes = plt.subplots(2,1,figsize=(7,7))
            axt0 = axes[0].twinx()
            axt1 = axes[1].twinx()
            for m_d,pst,label,color in zip(m_ds,psts,labels,colors):
                obs = pst.observation_data
                if oname not in obs.obsnme.values:
                    print("oname {0} missing from {1}".format(oname,m_d))
                    continue
                oe = pst.ies.obsen
                #print(oe.index.get_level_values(1))
                pr = oe.loc[oe.index.get_level_values(0)==0,:]
                pt = oe.loc[oe.index.get_level_values(0)==itr,:]
                #print(pr.shape,pt.shape)
                if pr.shape[0] == 0:
                    raise Exception("prior missing")
                if pt.shape[0] == 0:
                    continue
                #print(m_d)
                xvals = np.linspace(min(pr.loc[:,oname].values.min(),pt.loc[:,oname].values.min()),
                                    max(pr.loc[:,oname].values.max(),pt.loc[:,oname].values.max()),200)
                try:

                    kde = stats.gaussian_kde(pt.loc[:,oname].values)
                    yvals = kde(xvals)
                except:
                    yvals = np.array([np.nan for _ in xvals])
                
                
                if min(pr.loc[:,oname].values.min(),pr.loc[:,oname].values.min()) <=0.0:
                    
                    axt1.hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="none",hatch="X", edgecolor=color,density=True)
                    axes[1].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor=color,density=True)
                    axes[1].plot(xvals,yvals,color=color,lw=2.0,label=label)
                    axes[1].set_title(oname,loc="left")

                else:
                    axt1.hist(np.log10(pr.loc[:,oname].values),bins=20,alpha=0.5,facecolor="none",hatch="X", edgecolor=color,density=True)
                    axes[1].hist(np.log10(pt.loc[:,oname].values),bins=20,alpha=0.5,facecolor=color,density=True)
                    xvalslog = np.linspace(np.log10(xvals.min()),np.log10(xvals.max()),200)
                    try:
                        kde = stats.gaussian_kde(np.log10(pt.loc[:,oname].values))
                        yvalslog = kde(xvalslog)
                    except:
                        yvalslog = np.array([np.nan for _ in xvals])
                    axes[1].plot(xvalslog,yvalslog,color=color,lw=2.0,label=label)
                    axes[1].set_title(oname+" (log10)",loc="left")
                    ylim = axes[1].get_ylim()
                
                axt0.hist(pr.loc[:,oname].values,bins=20,alpha=0.5,facecolor="none",hatch="X", edgecolor=color,density=True)
                axes[0].hist(pt.loc[:,oname].values,bins=20,alpha=0.5,facecolor=color,density=True)
                axes[0].plot(xvals,yvals,color=color,lw=2.0,label=label)

                axes[0].set_title(oname,loc="left")
                axes[0].legend(loc="upper right")
                axes[1].legend(loc="upper right")
                
               
                #ylim = axes[0].get_ylim()
                #axes[0].plot([lower,lower],ylim,"k--",lw=2)
                #axes[0].plot([upper,upper],ylim,"k--",lw=2)
                axes[0].set_yticks([])
                axes[1].set_yticks([])
                axt0.set_yticks([])
                axt1.set_yticks([])
                ylim = axt0.get_ylim()
                axt0.set_ylim(ylim[0],ylim[1]*1.5)
                ylim = axt1.get_ylim()
                axt1.set_ylim(ylim[0],ylim[1]*1.5)
                
            
            
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
    return itr

def plot_summary_multiple(m_ds,labels,colors,figure_tag):
    
    assert len(m_ds) == len(labels)
    assert len(m_ds) == len(colors)

    
    iters = []
    arr_onames = []
    list_onames = []
    for m_d in m_ds:
        pst = pyemu.Pst(os.path.join(m_d,"pest.pst"),result_dir=m_d)
        print(pst.ies)
        iters.extend(pst.ies.phiactual.iteration.tolist())
        obs = pst.observation_data
        arrobs = obs.loc[obs.oname=="arrobs",:].copy()
        arrobs = arrobs.loc[arrobs.usecol=="pval"]
        arr_onames.extend(arrobs.obsnme.tolist())
        listobs = obs.loc[obs.oname=="listobs",:].copy()
        list_onames.extend(listobs.obsnme.tolist())
        dobs = obs.loc[obs.usecol.str.startswith("delay-head"),:]
        assert dobs.shape[0] > 0
        dobs["k"] = dobs.usecol.apply(lambda x: int(x.split(".")[1])) - 1
        uks = dobs.k.unique()
        

    list_onames = list(set(list_onames))
    list_onames.sort()
    arr_onames = list(set(arr_onames))
    arr_onames.sort()

    iters = list(set(iters))
    iters.sort()

    if 0 in iters:
        iters.remove(0)  
    from multiprocessing import Pool

    pool = Pool(processes=5)
    results = []  
    for itr in iters:
        #_plot_compare(itr,uks,arr_onames,list_onames,figure_tag,m_ds,psts,labels,colors)
        #break
        r = pool.apply_async(_plot_compare,args=(itr,uks,arr_onames,list_onames,figure_tag,m_ds,labels,colors))
        results.append(r)
    for r in results:
        print(r.get())
    pool.close()
    pool.join()


        

def get_master_dirs(tags=None,sites=None,exclude=None,master_tag="master"):
    skip = ["dependencies","bin","csub_example","etc","thmodel",".git"]
    if tags is not None:
        if not isinstance(tags,list):
            tags = [tags]
    site_dirs = [d for d in os.listdir(".") if os.path.isdir(d) and not d.lower() in skip]
    if sites is not None:
        site_dirs = [d for d in site_dirs if d.lower() in sites]
    m_ds = {}
    for site_dir in site_dirs:
        sdirs = [os.path.join(site_dir,d) for d in os.listdir(site_dir) if os.path.isdir(os.path.join(site_dir,d))]
        if master_tag is not None:
            sdirs = [d for d in sdirs if master_tag in d.lower()]
        if tags is not None:

            sdirs = [d for d in sdirs if tag in d.lower()]
        if len(sdirs) > 0:
            for m_d in sdirs:
                bunk = os.path.join(os.path.join(m_d,"pest.obs+noise.csv"))
                print(bunk)
                if os.path.exists(bunk):
                    print("removing")
                    os.remove(bunk)
            m_ds[site_dir] = sdirs

    return m_ds
    

def md_2_label(m_d):
    label = ""
    if "nofocus" in m_d:
        label += "no focus,"
    else:
        label += "focus,"
    if "nospecstate" in m_d:
        label += "no spec state,"
    else:
        label += "spec state,"
    if not m_d.endswith("state"):
        label += m_d.split("_")[-1]
    return label

def process_compaction_tdiff_obs(t_d="."):
    sdf = pd.read_csv(os.path.join(t_d,"datetime_model.csub.obs.csv"),index_col=0,parse_dates=True)
    sdf = sdf.loc[:,[c for c in sdf.columns if c.startswith("compaction.")]]    
    tdiffs = {}
    for col in sdf.columns:
        vals = sdf.loc[:,col].values
        diff = vals[1:] - vals[:-1]
        tdiffs["tdif-"+col] = diff

    difdf = pd.DataFrame(tdiffs,index=sdf.index[1:])
    difdf.index.name = "datetime"
    difdf.to_csv(os.path.join(t_d,"compaction_tdiffs.csv"))
    return difdf


def process_compaction_at_above_obs(t_d="."):
    sdf = pd.read_csv(os.path.join(t_d,"datetime_model.csub.obs.csv"),index_col=0,parse_dates=True)
    sdf = sdf.loc[:,[c for c in sdf.columns if c.startswith("compaction.")]]  
    sdf.sort_index(inplace=True,axis=1)

    layers = [int(c.split(".")[1]) for c in sdf.columns]
    data = {}
    for k,lay in enumerate(layers):
        data["compactatabove.{0}".format(lay)] = sdf.iloc[:,:k+1].values.sum(axis=1)
    
    adf = pd.DataFrame(data=data,index=sdf.index)
    adf.index.name = "datetime"
    adf.to_csv(os.path.join(t_d,"compaction_atabove.csv"))
    
    return adf

def process_delay_obs(t_d="."):
    sdf = pd.read_csv(os.path.join(t_d,"datetime_model.csub.obs.csv"),index_col=0,parse_dates=True)
    gdf = pd.read_csv(os.path.join(t_d,"datetime_model.gwf.obs.csv"),index_col=0,parse_dates=True)    
    ddf = sdf.loc[:,sdf.columns.str.contains("delay-head")]
    vals = ddf.values
    vals[vals==-999.0] = np.nan
    lowest = []
    for i in range(ddf.shape[0]):
        lowest.append(ddf.iloc[:i+1,:].min())
    lowest = pd.DataFrame(lowest,columns=ddf.columns,index=ddf.index)
    ddf_layers = [c.split(".")[1] for c in ddf.columns]
    diffs = {}
    for col in ddf.columns:
        layer = col.split(".")[1]
        gcol = [c for c in gdf.columns if c.endswith(".{0}".format(layer))]
        assert len(gcol) == 1,str(gdf.columns)
        diffs["gwf-minus-delay.{0}".format(layer)] = gdf.loc[:,gcol].values.flatten() - ddf.loc[:,col].values.flatten()
        diffs["gwf-minus-delaylowest.{0}".format(layer)] = gdf.loc[:,gcol].values.flatten() - lowest.loc[:,col].values.flatten()    
    
    difdf = pd.DataFrame(diffs,index=gdf.index)
    difdf.fillna(-999.0,inplace=True)
    difdf.to_csv(os.path.join(t_d,"delay_diffs.csv"))
    return difdf
    

def extract_delay_results(m_d):
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"),result_dir=m_d)
    oe = pst.ies.obsen
    itrs = oe.index.get_level_values(0).unique().values
    itrs.sort()
    obs = pst.observation_data
    print(obs.oname.unique())
    dobs = obs.loc[obs.oname=="delaydif",:].copy()
    dobs["datetime"] = pd.to_datetime(dobs.datetime)
    grps = dobs.usecol.unique()
    grps.sort()
    with PdfPages(os.path.join(m_d,"delaydif_summary.pdf")) as pdf:
        fig,axes = plt.subplots(len(grps),1,figsize=(11,8.5))
        for igrp,grp in enumerate(grps):
            dfs = []
            for itr in itrs:
                gobs = dobs.loc[dobs.usecol==grp,:].copy()
                gobs.sort_values(by="datetime",inplace=True)  
                df = oe.loc[oe.index.get_level_values(0)==itr,gobs.obsnme]
                df = df.droplevel(0)
                
                df.columns = gobs.datetime.values
                df = df.T
                df.index.name = "datetime"
                #csv_file = os.path.join(m_d,"{0}.{1}.csv".format(grp.replace(":","-"),itr))
                #df.to_csv(csv_file)
                vals = df.values
                vals[vals==-999.] = np.nan
                dfs.append(df)
                print(grp,itr)
                

            df = pd.concat(dfs,keys=itrs)
            #print(df)
            df.index.names = ["iteration","datetime"]
            df.to_csv(os.path.join(m_d,grp.replace(":","-")+".csv"))
            
            ax = axes[igrp]
            cmap = plt.get_cmap("viridis")
            norm = matplotlib.colors.Normalize(vmin=itrs.min(),vmax=itrs.max())
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm) 
            for df,itr in zip(dfs,itrs):
                dts = df.index.values
                vals = df.values
                [ax.plot(dts,vals[:,j],alpha=itr/itrs.max(),color=cmap(norm(itr))) for j in range(vals.shape[1])]
            ax.set_ylim(np.nanmin(vals)-np.nanstd(vals),np.nanmax(vals)+np.nanstd(vals))
            plt.colorbar(sm,ticks=[itrs.min(),np.median(itrs),itrs.max()],label="iteration",ax=ax)
            ax.grid()
            ax.set_title(grp,loc="left")
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)

        exit()
    #dobs = obs.loc[obs]


def try_run_modify_pst_fxn(site_name,tpl_dir):
    sys.path.insert(0,site_name)
    import prep_data
    reload(prep_data)
    assert site_name in prep_data.__file__
    lines = open(os.path.join(site_name,"prep_data.py")).readlines()
    has_modpst = False
    for line in lines:
        if line.startswith("def modify_pst("):
            has_modpst = True
            break
    if has_modpst:
        if not callable(getattr(prep_data,'modify_pst')):
            raise Exception("modify_pst() not callable")
        #first save all the org values
        pst = pyemu.Pst(os.path.join(tpl_dir,"pest.pst"))
        par = pst.parameter_data
        for col in ["parval1","parlbnd","parubnd"]:
            par[col+"_org"] = par[col].copy()
        obs = pst.observation_data
        for col in ["obsval","weight","obgnme"]:
            obs[col+"_org"] = obs[col].copy() 
        org_nnz_names = pst.nnz_obs_names
        pst.write(os.path.join(tpl_dir,"pest.pst"),version=2)   
        prep_data.modify_pst(tpl_dir)
        pst = pyemu.Pst(os.path.join(tpl_dir,"pest.pst"))
        missing = list(set(pst.nnz_obs_names) - set(org_nnz_names))
        if len(missing) > 0 and "ies_obs_en" in pst.pestpp_options:
            print("newly non-zero weighed obs detected, drawing noise for them...")
            noise = pyemu.ObservationEnsemble.from_binary(pst=pst,
                        filename=os.path.join(tpl_dir,pst.pestpp_options["ies_obs_en"]))
            new_noise = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst,
                       cov=pyemu.Cov.from_observation_data(pst),
                       num_reals=noise.shape[0])
            new_noise.loc[:,noise.columns] = noise.values
            new_noise.to_binary(os.path.join(tpl_dir,pst.pestpp_options["ies_obs_en"]))

        pst.control_data.noptmax = -2
        pst.write(os.path.join(tpl_dir,"pest.pst"),version=2) 
        pyemu.os_utils.run("pestpp-ies pest.pst",cwd=tpl_dir)

def process_percent_comp_obs(t_d="."):
    infile = os.path.join(t_d,"datetime_model.csub.obs.csv")
    outfile = os.path.join(t_d,"percent_compaction.csv")
    df = pd.read_csv(infile,index_col=0,parse_dates=True)

    df = df.loc[:,df.columns.map(lambda x: x.startswith("compaction."))]
    vals = df.values
    vals[vals<1e-6] = 0.0 # only count postive compaction (no rebound) and not noisey tiny values
    df.loc[:,:] = vals
    df.sort_index(axis=1,inplace=True)

    assert df.shape[1] > 0
    total = df.sum(axis=1).values
    df = df.mul(1/total,axis=0)
    df *= 100.0
    df.columns = ["percentcomp."+c.split(".")[1] for c in df.columns]
    df.fillna(0,inplace=True)
    df.to_csv(outfile)
    return df


if __name__ == "__main__":
    t_d = os.path.join("PORT","template_ies_delay_y_nofocus_nospecstate_thickcontrol4")
    process_compaction_at_above_obs(t_d)
    exit()
    #extract_delay_results(os.path.join("D454","master_ies_delay_y_nofocus_specstate_newpyemu"))
    #m_d = os.path.join("H201","master_ies_delay_y_nofocus_nospecstate_test4")
    #t_d = os.path.join("H201","template_ies_delay_y_nofocus_nospecstate_test4")
    #xfer_from_m_d(t_d,m_d)
    #exit()
    #t_d = os.path.join("j88","template_ies_delay_y_nofocus_nospecstate_lessnoise")
    #process_delay_obs(t_d)
    #plot_en_compaction(os.path.join("H201","master_ies_delay_y_nofocus_nospecstate_reweight3_newbin_70percent"),noptmax="combined")
    #exit()
    #m_d = os.path.join("GWM_14","master_ies_delay_y_nofocus_nospecstate_wghb_ndelayobs_nolessthan_newbin_noreinf_morris")
    #t_d = os.path.join("GWM_14","/Users/jwhite/Dev/1D_CSUB/GWM_14/template_ies_delay_y_nofocus_nospecstate_wghb_ndelayobs_nolessthan_newbin_htmin_noreinf")

    #process_percent_comp_obs(t_d)
    #exit()
    m_d = os.path.join("F286","master_ies_delay_y_nofocus_nospecstate_newobs2_70percent2024_strong_wmorris_morris")
    plot_morris_delaydif_summary(m_d)
    exit()
    labels = ["70% pref","no pref"]
    colors = ["m","g"]
    m_ds = [os.path.join("F286","master_ies_delay_y_nofocus_nospecstate_newobs2_70percent2024_strong"),
            os.path.join("F286","master_ies_delay_y_nofocus_nospecstate_newobs2")]
    #plot_all(m_ds[0],plot_all_iters=True)
    #exit()
    plot_summary_multiple(m_ds,labels,colors,"test")
    exit()
    m_d_dict = get_master_dirs(sites=["j88"])
    sites = list(m_d_dict.keys())
    sites.sort()
    print(sites)
    
    for site in sites:
        m_ds = m_d_dict[site]
        print(m_ds)
        labels = [site+" "+md_2_label(m_d) for m_d in m_ds]
        colors = ["m","c"]
        plot_summary_multiple(m_d_dict[site],labels,colors,"test")


    # site_name = "EARLIMART"
    # m_d = os.path.join(site_name, "master_ies_delay_m_nofocus_20241217")
    # plot_scenarios(site_name=site_name, m_d=m_d)
    # plot_WL_SUB_2_panel(site_name=site_name, m_d=m_d, noptmax=None)
    # plot_all(site_name=site_name, master_dir=m_d, plot_all_iters=False)
    # plot_all(site_name=site_name, master_dir=m_d, plot_all_iters=True)
    # plot_en_subsidence_scenarios(site_name=site_name, m_d=m_d, new_m_d=m_d+"_2015_scenario",
    #                              plot_obs=False, noptmax=None)

    #tpl_dir = os.path.join("J88","template_delay_m_nofocus_nospecstate6")
    #xfer_m_d = os.path.join("J88","master_ies_delay_y_nofocus_nospecstate6")
    #xfer_from_m_d(tpl_dir,xfer_m_d)



    #plot_window_compare("results",tag="annual")
    #plot_window_compare("results", tag="monthly")
    #plot_en_compaction(os.path.join("J88","master_ies_delay_focus_annualtest"))
    #exit()

    # for site in ["J88","T88"]:
    #     m_d1 = os.path.join(site,"master_ies_delay_focus_annualdelay")
    #     m_d2 = m_d1.replace("delay","nodelay")
    #     #plot_all(m_d1,plot_all_iters=True)
    #     #plot_all(m_d2,plot_all_iters=True)
    #     #plot_compare_parameter_summary([m_d1,m_d2],noptmax="combined")
    #     plot_compare_sub([m_d1,m_d2],noptmax="combined")
        
    #exit()
    #     m_d = os.path.join(site,"master_ies_nodelay_focus_annualnodelay")
    #     plot_all(m_d,plot_all_iters=True)
    # exit()
    # site = "J88"
    # m_d = os.path.join(site,"master_ies_delay_focus_annualdelay")
    
    # run_local(os.path.join(site,"template_annualdelay"), None,
    #           pst_name="pest.pst", num_workers=20, port=4004)
    # exit()
    
    #plot_en_compaction(m_d, noptmax=None)
    #exit()

    # morris_mds = []
    # for site in ["J88","T88"]:
    #     #run_morris(os.path.join(site,"template_annualnodelay_scenarios"))
    #     #run_morris(os.path.join(site,"template_annualdelay_scenarios"))
    #     m_ds = [os.path.join(site,m_d) for m_d in os.listdir(site) if os.path.isdir(os.path.join(site,m_d)) and "morris" in m_d and "master" in m_d]
    #     assert len(m_ds) > 0
    #     morris_mds.extend(m_ds)
    # plot_morris_compare(morris_mds)
    # exit()
    #site = "376.676"
    #run_scenarios(site,os.path.join(site,"template_monthly"),
    #              os.path.join(site,"master_ies_delay_focus_monthly"),
    #              run_prior_scenarios=False)

    #make_combined_posterior(os.path.join(site,"master_ies_delay_nofocus_annualtest_addobs"))
    #plot_en_compaction_scenarios(os.path.join(site,"master_ies_delay_focus_annualtestnodelay_scenarios_prior"),bayes_stance="pr")
    #run_morris(os.path.join("D454","template_test_specstate_scenarios"),scenario_name="mtnomo")
    #plot_morris(os.path.join("D454","master_test_specstate_scenarios_morris_mtnomo"))
    #exit()
    #t_d = os.path.join("J88","templatemonthly")
    #test_csv_to_ghb(t_d)
    #location = "T88"
    #m_d = os.path.join(location,"master_ies_delay_focusnewhdsobs")
    #run_scenarios(location,org_t_d,m_d,subset_size=10)
    #plot_en_compaction_scenarios(m_d+"_scenarios")
    #export_realizations_to_dirs(org_t_d,m_d+"_scenarios",real_name_tag="real:base")

    #plot_en_compaction(m_d)
    #t_d = os.path.join("376.676","template_delay_m_nofocus_nospecstate5_nodiff_noxfer")
    #set_obsvals_and_weights(os.path.join("376.676","processed_data","376.676_sub_data.csv"),t_d)
    #plot_par_summary(m_d)
    #plot_model_input_summary(m_d)
    #plot_model_input_summary_ht_compare(os.path.join('T949R','master_ies_delay_focusann_min'),noptmax=5)
    #plot_compare_both_ht(os.path.join('T949R','master_ies_delay_focusann_min'))
    #plot_compare_both_ht(os.path.join('D454','master_ies_delay_focus_ann_min'))


    #setup_diff_obs(os.path.join("J88","template"))
    #plot_all(os.path.join("U822","master_ies_delay_focusannual"),plot_all_iters=True)
    #exit()
    #m_d = os.path.join("J88","master_ies_delay_nofocus")
    #t_d = os.path.join("J88","template")
    #prep_for_hypoth_test(t_d,m_d)
    #exit()
    #t_d_ht = t_d + "_ht"
    #prep_for_parallel(t_d_ht, t_d_ht+"_clean", noptmax=10,
    #                  num_reals=None, overdue_giveup_fac=10.0,
    #                  overdue_giveup_time=10.0)

    #m_d_ht = m_d + "_ht"
    #run_local(t_d_ht+"_clean", m_d_ht, pst_name="pest.pst", num_workers=10)
    #plot_compare_ht(m_d,m_d_ht)
