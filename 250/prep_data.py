import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

name_k_dict = {"upper": 0, "lower": 1}

from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join("..", "dependencies"))

def prep_data(use_delay, **kwargs):
    from dependencies.project_functions.interbed_functions import interbed_thicknesses
    from dependencies.project_functions.layer_functions import layer_thicknesses

    w_d = "."
    location = "250"
    if os.path.exists("processed_data"):
        shutil.rmtree("processed_data")
    os.makedirs("processed_data")

    par_data = pd.read_excel(
        os.path.join(w_d, "source_data", "{0}_par_data.xlsx".format(location))
    )
    par_data = par_data.loc[:, ["parameter", "Upper", "Corcoran", "Lower"]]
    par_data.index = par_data.pop("parameter").values
    par_data.dropna(inplace=True)

    obs = pd.read_csv(
        os.path.join(w_d, "source_data", "{0}_obs_data.csv".format(location)),
        index_col=0,
        parse_dates=True,
    )
    lith = (
        pd.read_csv(
            os.path.join(w_d, "source_data", "{0}_lithology.csv".format(location))
        )
        .dropna(axis=1, how="all")
        .dropna(axis=0)
    )
    lith.Aquifer = lith.Aquifer.str.lower()
    lay_df = pd.DataFrame(columns=lith.Aquifer.unique())
    nlay = 3

    k_vals = [10.0 for _ in range(nlay)]
    k33_vals = [0.01 for _ in range(nlay)]

    interpolated_obs_dfs = []
    uaq = ["Upper", "Lower"]
    uk = [0, 2]
    top = None
    for k, aq in zip(uk, uaq):
        aobs = obs.loc[obs.Aquifer == aq, ["Alt"]].copy()
        print(aobs.loc[aobs.index.duplicated()])
        if aobs.loc[aobs.index.duplicated()].shape[0] > 0:
            aobs = aobs.groupby(aobs.index).min()
        dt_range = pd.date_range(aobs.index.min(), aobs.index.max(), freq="d")
        aobs_interp = aobs.reindex(dt_range)
        aobs_interp["interpolated"] = aobs_interp.Alt.interpolate(method="time")
        if "wl_func" in kwargs:
            aobs_interp["interpolated"] = kwargs["wl_func"](aobs_interp["interpolated"])
        aobs_interp["Aquifer"] = aq
        aobs_interp["klayer"] = k
        interpolated_obs_dfs.append(aobs_interp)
        # if k == 0:
        # aobs = obs.loc[obs.Aquifer==aq,:]
        # top = (aobs.BLS + aobs.Alt).max()
    # if top is None:
    top = (obs.Alt).max()

    tsdf = pd.concat(interpolated_obs_dfs)

    fig, axes = plt.subplots(len(uaq), 1, figsize=(10, 10))

    for ax, aq in zip(axes, uaq):
        if aq == "Composite":
            continue
        uobs = obs.loc[obs.Aquifer == aq, :].copy()
        uobs.sort_index(inplace=True)
        print(aq, obs.columns, tsdf.columns)
        ax.scatter(uobs.index, uobs["Alt"], marker=".", c="r", label="raw", alpha=0.5)
        ax.plot(uobs.index, uobs["Alt"], "r-", label="raw", alpha=0.5)
        pobs = tsdf.loc[tsdf.Aquifer == aq, :].copy()
        pobs.sort_index(inplace=True)
        # ax.scatter(uobs.index,uobs.Alt,marker='o',s=20,c='m',alpha=0.5,label="processed")
        ax.plot(pobs.index, pobs.interpolated.values, "m-", label="processed")
        ax.set_title(aq, loc="left")
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(w_d, "processed_data", "processed_gwlevels.pdf"))
    plt.close(fig)
    tsdf.to_csv(os.path.join(w_d, "processed_data", "{0}.ts_data.csv".format(location)))

    cg_theta = [0.3 for _ in range(nlay)]
    cg_ske_cr = par_data.loc["cg_ske_cr", :].values
    ssv_cc = par_data.loc["ssv_cc", :].values
    sse_cr = par_data.loc["sse_cr", :].values
    ib_kv = par_data.loc["ib_kv", :].values
    sgm = 1.7
    sgs = 2.0
    theta = 0.35
    h0 = 0.0

    if use_delay:
        lay_df.loc["cdelay", :] = ["nodelay"] + [
            "delay" for _ in range(len(lay_df.columns) - 1)
        ]
    else:
        lay_df.loc["cdelay", :] = "nodelay"

    # top = (obs.BLS + obs.Alt).max()  # ?  what is going here???
    # set dataframe entries that are known at this point
    lay_df.loc["pcs0", :] = par_data.loc["pcs0", :].values
    lay_df.loc["h0", :] = h0
    lay_df.loc["ssv_cc", :] = ssv_cc
    lay_df.loc["sse_cr", :] = sse_cr
    lay_df.loc["theta", :] = theta
    lay_df.loc["kv", :] = ib_kv
    lay_df.loc["cg_theta", :] = cg_theta
    lay_df.loc["cg_ske_cr", :] = cg_ske_cr
    lay_df.loc["aquifer_name", :] = lay_df.columns
    lay_df.loc["sgm", :] = sgm
    lay_df.loc["sgs", :] = sgs
    lay_df.loc["k", :] = k_vals
    lay_df.loc["k33", :] = k33_vals

    # define botm
    lay_df = layer_thicknesses(lay_df, lith, top)

    # interbed thickness data
    quantiles = kwargs.pop("quantiles", None)
    lay_df = interbed_thicknesses(lay_df, lith, quantiles, use_delay)

    lay_df.columns = np.arange(nlay, dtype=int)
    lay_df.index.name = "property"
    print(lay_df)

    lay_df.to_csv(
        os.path.join(w_d, "processed_data", f"{location}.model_property_data.csv")
    )

def modify_pst(tpl_dir):
    import pyemu
    location = "250"
    pst = pyemu.Pst(os.path.join( tpl_dir, "pest.pst"))
    obs = pst.observation_data

    cobs = obs.loc[obs.usecol.str.startswith("percentcomp."), :].copy()
    cobs["layer"] = cobs.usecol.apply(lambda x: int(x.split(".")[1]))
    cobs["datetime"] = pd.to_datetime(cobs.datetime)
    cobs3 = cobs.loc[
            cobs.apply(lambda x: x.datetime.year == 2024 and x.layer == 3, axis=1), :
            ]
    assert cobs3.shape[0] > 0
    cobs3.sort_values(by="datetime", inplace=True)
    obs.loc[cobs3.index, "obsval"] = 70
    obs.loc[cobs3.index, "standard_deviation"] = 2.5
    obs.loc[cobs3.index, "obgnme"] = "prefer-compact3"
    obs.loc[cobs3.index, "weight"] = 100.0

    phi_file = pst.pestpp_options.get("ies_phi_factor_file", None)
    if phi_file is not None:
        df = pd.read_csv(
            os.path.join(tpl_dir, phi_file), header=None, names=["tag", "prop"]
        )

        if "less_than_rebound" not in df.tag.values:
            df.index = df.pop("tag")
            df.loc["prefer-compact3", "prop"] = 1
            df.to_csv(os.path.join( tpl_dir, phi_file), header=False)

    pst.control_data.noptmax = -2
    pst.write(os.path.join(tpl_dir, "pest.pst"), version=2)

def clamp_earlytime_ghb_bounds(tpl_dir):
    """
    Narrow bounds for early-time GHB head parameters for this site.

    Parameters
    ----------
    pst_path : str
        Path to pest.pst in the site’s template directory (e.g., './template/pest.pst')
    year_cutoff : int
        Any GHB parameter tagged with a datetime < Jan 1 of this year is considered “early”
    margin_ft : float
        +/- band around the current parval1 used to set new bounds
    """
    import pyemu
    import re
    w_d = "."
    location = "250"

    year_cutoff = 1944
    margin_ft = 20.0

    pst = pyemu.Pst(os.path.join(tpl_dir, "pest.pst"))
    pdf = pst.parameter_data

    ghb_mask = pdf.parnme.str.contains("ghbhead", case=False, na=False)

    # Pull datetimes from parnme (YYYY-MM-DD); skip any that don’t have a date
    date_pat = re.compile(r"(\d{4}-\d{2}-\d{2})")
    par_dates = pd.to_datetime(
        pdf.loc[ghb_mask, "parnme"].str.extract(date_pat, expand=False),
        errors="coerce"
    )

    early_mask = ghb_mask.copy()
    early_mask.loc[ghb_mask] = par_dates < pd.Timestamp(f"{year_cutoff}-01-01")

    # tighten bounds around current parval1
    pv = pdf.loc[early_mask, "parval1"]
    pdf.loc[early_mask, "parlbnd"] = pv - margin_ft
    pdf.loc[early_mask, "parubnd"] = pv + margin_ft

    # (optional) shift the starting values (parval1) if you want a bias
    # pdf.loc[early_mask, "parval1"] = pdf.loc[early_mask, "parval1"].clip(pv - 0.5*margin_ft, pv + 0.5*margin_ft)

    pst.parameter_data = pdf
    pst.control_data.noptmax = -2
    pst.write(os.path.join( tpl_dir, "pest.pst"), version=2)
    print(f"Tightened early-time GHB bounds for {early_mask.sum()} parameters "
          f"(< {year_cutoff}) to ±{margin_ft} ft.")

if __name__ == "__main__":
    os.chdir("250")
    prep_data(True)
