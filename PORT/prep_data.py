import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join("..", "dependencies"))

def prep_data(use_delay, use_shitty_gsi_data=False, **kwargs):
    from dependencies.project_functions.interbed_functions import interbed_thicknesses
    from dependencies.project_functions.layer_functions import layer_thicknesses

    w_d = "."
    location = "PORT"

    if os.path.exists("processed_data"):
        shutil.rmtree("processed_data")
    os.makedirs("processed_data")

    par_data = pd.read_excel(
        os.path.join(w_d, "source_data", "{0}_par_data.xlsx".format(location))
    )
    par_data = par_data.loc[
        :, ["parameter", "Upper", "Lower", "Pliocene", "Santa Margarita"]
    ]
    par_data.index = par_data.pop("parameter").values

    obs = pd.read_csv(
        os.path.join(w_d, "source_data", "{0}_obs_data.csv".format(location)),
        index_col=0,
        parse_dates=True,
    )
    if use_shitty_gsi_data is True:
        gsidf = pd.read_csv(os.path.join(w_d, "source_data", "GWL.csv"))

        gsidf["year"] = gsidf.Time.apply(
            lambda x: "19" + x.split("-")[2]
            if int(x.split("-")[2]) > 25
            else "20" + x.split("-")[2]
        )
        gsidf["datetime"] = gsidf.apply(
            lambda x: x.Time.split("-")[0] + "-" + x.Time.split("-")[1] + "-" + x.year,
            axis=1,
        )
        gsidf.index = pd.to_datetime(gsidf.pop("datetime"))
        gsidf.pop("Time")
        gsidf.pop("year")
        gsidf.columns = ["BLS"]

        top = obs.Alt.max()

        gsidf["Alt"] = top - gsidf.BLS
        gsidf["BLS"] = np.nan  # so that it does change the top calc below...
        if "gsi_2_all_layers" in kwargs and kwargs["gsi_2_all_layers"] is True:
            uaq = obs.Aquifer.unique()
            dfs = []
            for aq in uaq:
                df = gsidf.copy()
                df["Aquifer"] = aq
                dfs.append(df)
            obs = pd.concat(dfs)
            # print(obs)
            # exit()
        else:
            gsidf["Aquifer"] = "Santa Margarita"
            obs = obs.loc[obs.Aquifer != "Santa Margarita"]
            obs = pd.concat([obs, gsidf])

    else:
        obs.loc[-1] = [np.nan, 30, "Santa Margarita"]
        ivals = obs.index.tolist()
        ivals[-1] = pd.to_datetime("1-1-2015")
        obs.index = ivals

    lith = (
        pd.read_csv(
            os.path.join(w_d, "source_data", "{0}_lithology.csv".format(location))
        )
        .dropna(axis=1, how="all")
        .dropna(axis=0)
    )
    lith.Aquifer = lith.Aquifer.str.lower()
    lay_df = pd.DataFrame(columns=lith.Aquifer.unique())
    nlay = len(lith.Aquifer.unique())

    k_vals = [10.0 for _ in range(nlay)]
    k33_vals = [0.01 for _ in range(nlay)]

    interpolated_obs_dfs = []
    interp_org_dfs = []
    uaq = ["Lower", "Pliocene", "Santa Margarita"]
    uk = [1, 2, 3]
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
        interpolated_obs_dfs.append(aobs_interp.copy())

        if k == 0:
            aobs = obs.loc[obs.Aquifer == aq, :]
            top = (aobs.BLS + aobs.Alt).max()
        aobs_interp = aobs_interp.loc[aobs.index, :]
        interp_org_dfs.append(aobs_interp)
    if top is None:
        top = (obs.BLS + obs.Alt).max()

    tsdf = pd.concat(interpolated_obs_dfs)
    tsdf_org = pd.concat(interp_org_dfs)
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
        pobs = tsdf_org.loc[tsdf_org.Aquifer == aq, :].copy()
        pobs.sort_index(inplace=True)
        print(pobs)
        # ax.scatter(uobs.index,uobs.Alt,marker='o',s=20,c='m',alpha=0.5,label="processed")
        ax.scatter(
            pobs.index,
            pobs.interpolated.values,
            marker="o",
            facecolor="none",
            edgecolor="c",
            s=50,
            label="processed sample to org",
        )
        ax.set_title(aq, loc="left")
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(w_d, "processed_data", "processed_gwlevels.pdf"))
    plt.close(fig)
    tsdf.to_csv(os.path.join(w_d, "processed_data", "{0}.ts_data.csv".format(location)))
    tsdf_org.to_csv(
        os.path.join(w_d, "processed_data", "{0}.orgts_data.csv".format(location))
    )

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

    top = (obs.Alt).max()  # ?  what is going here???

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

    subdf = pd.read_csv(
        os.path.join(w_d, "source_data", "{0}_sub_data.csv".format(location)),
        index_col=0,
        parse_dates=True,
    )
    subdf["Source"] = subdf.Source.str.lower()
    esubdf = subdf.loc[subdf.Source.str.contains("ext"), :]
    assert len(esubdf) > 0
    nesubdf = subdf.loc[~subdf.Source.str.contains("ext"), :]
    assert len(nesubdf) > 0
    esubdf.to_csv(
        os.path.join(w_d, "processed_data", "{0}_extsub_data.csv".format(location))
    )
    nesubdf.to_csv(
        os.path.join(w_d, "processed_data", "{0}_sub_data.csv".format(location))
    )

def modify_pst(tpl_dir):
    import pyemu

    pst = pyemu.Pst(os.path.join(tpl_dir, "pest.pst"))
    obs = pst.observation_data
    site = tpl_dir.split(os.path.sep)[0]

    obs_file = os.path.join(site, "processed_data", "PORT_extsub_data.csv")
    odf = pd.read_csv(obs_file)
    odf.columns = [c.lower() for c in odf.columns]
    odf["datetime"] = pd.to_datetime(odf.pop("date"))
    odf["source"] = odf.source.str.lower()
    odf.sort_values(by="datetime", inplace=True)

    sobs = obs.loc[obs.usecol == "compactatabove.3", :].copy()
    assert sobs.shape[0] > 0
    obs.loc[sobs.obsnme, "weight"] = 0.0
    obs.loc[sobs.obsnme, "observed"] = False
    obs.loc[sobs.obsnme, "standard_deviation"] = np.nan

    sobs["datetime"] = pd.to_datetime(sobs.datetime, format="%Y-%m-%d")
    sobs.sort_values(by="datetime", inplace=True)
    udts = np.unique(sobs.datetime.values)
    udts.sort()
    print(udts)
    # exit()
    datetimes, sub, source, count = [], [], [], []
    start_datetimes, org_datetimes = [], []
    should_assimilate = []

    for start, end in zip(udts[:-1], udts[1:]):
        udf = odf.loc[odf.datetime.apply(lambda x: x >= start and x < end), :].copy()

        if udf.shape[0] == 0:
            continue
        print(udf)
        print(start, end)
        datetimes.append(end)
        sub.append(udf["subsidence_ft"].max())
        print(sub)
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
        org_datetimes.append(
            "-".join([dt.strftime("%Y%m%d") for dt in udf.datetime.tolist()])
        )

    deltas = (sobs["datetime"].iloc[1:] - sobs["datetime"].iloc[:-1]).dt.days
    odf = pd.DataFrame(
        data={
            "datetime": datetimes,
            "compactatabove.3": sub,
            "source": source,
            "count": count,
            "start_datetime": start_datetimes,
            "org_datetimes": org_datetimes,
            "should_assimilate": should_assimilate,
        },
        index=datetimes,
    )
    odf["org_compactatabove.3"] = odf["compactatabove.3"].copy()
    # odf["compactatabove.3"] -= odf["compactatabove.3"].min() #reset to 0.0 sub at the start
    odf.to_csv(os.path.join(tpl_dir, "sampled_extobs.csv"))
    deltas = np.array(deltas)

    count = 0
    verf_count = 0
    skipped = 0
    too_far = []
    sub_max = -1e30
    names = []
    for dt, subval, src, should_assimilate in zip(
        odf.datetime, odf["compactatabove.3"], odf.source, odf.should_assimilate
    ):
        sobs["distance"] = (dt - sobs.datetime).dt.days.apply(np.abs)
        if sobs.distance.min() > (deltas.mean() * 1.5):
            print("...sub datum too far to interpolate", dt, subval, src)
            too_far.append([dt, subval, src])
            continue
        sobs.sort_values(by="distance", inplace=True)
        # print(dt,sobs.distance.describe())

        idxmin = sobs.obsnme.iloc[0]
        mobs = sobs.loc[[idxmin], :]
        assert mobs.shape[0] == 1
        oname = mobs.obsnme.values[0]
        sub_max = max(subval, sub_max)
        if obs.loc[oname, "observed"] == True:
            print(oname)
            print("duplicate obs for datetime: {0}".format(dt))
            continue
        obs.loc[oname, "observed"] = True
        obs.loc[oname, "obsval"] = subval
        obs.loc[oname, "source"] = src
        if should_assimilate:
            obs.loc[oname, "weight"] = 1.0
        else:
            # obs.loc[oname,"weight"] = 0.25
            skipped += 1
        obs.loc[oname, "lower_bound"] = 0.0

        obs.loc[oname, "standard_deviation"] = 0.05  # max(0.0001,subval * 0.25)
        count += 1
        names.append(oname)

    print(count)
    assert count > 0
    print(obs.loc[names, "obsval"])

    cobs = obs.loc[obs.usecol.str.startswith("percentcomp."), :].copy()
    obs.loc[cobs.obsnme, "weight"] = 0.0

    cobs["layer"] = cobs.usecol.apply(lambda x: int(x.split(".")[1]))
    cobs["datetime"] = pd.to_datetime(cobs.datetime)
    cobs1 = cobs.loc[
        cobs.apply(lambda x: x.datetime.year == 2024 and x.layer == 1, axis=1), :
    ]
    cobs2 = cobs.loc[
        cobs.apply(lambda x: x.datetime.year == 2024 and x.layer == 2, axis=1), :
    ]
    cobs3 = cobs.loc[
        cobs.apply(lambda x: x.datetime.year == 2024 and x.layer == 3, axis=1), :
    ]
    cobs4 = cobs.loc[
        cobs.apply(lambda x: x.datetime.year == 2024 and x.layer == 4, axis=1), :
    ]
    assert cobs1.shape[0] > 0
    cobs1.sort_values(by="datetime", inplace=True)
    assert cobs2.shape[0] > 0
    cobs2.sort_values(by="datetime", inplace=True)
    assert cobs3.shape[0] > 0
    cobs3.sort_values(by="datetime", inplace=True)
    assert cobs4.shape[0] > 0
    cobs4.sort_values(by="datetime", inplace=True)
    # obs.loc[cobs1.index, "obsval"] = 1
    # obs.loc[cobs1.index, "obgnme"] = "greater_than-compact1"
    # obs.loc[cobs1.index, "weight"] = 100.0
    # obs.loc[cobs2.index, "obsval"] = 1
    # obs.loc[cobs2.index, "obgnme"] = "greater_than-compact2"
    # obs.loc[cobs2.index, "weight"] = 100.0
    obs.loc[cobs3.index, "obsval"] = 15
    obs.loc[cobs3.index, "obgnme"] = "greater_than-compact3"
    obs.loc[cobs3.index, "weight"] = 100.0
    obs.loc[cobs4.index, "obgnme"] = "greater_than-compact4"
    obs.loc[cobs4.index, "weight"] = 100.0
    obs.loc[cobs4.index, "obsval"] = 15

    phi_file = pst.pestpp_options.get("ies_phi_factor_file", None)
    if phi_file is not None:
        df = pd.read_csv(
            os.path.join(tpl_dir, phi_file), header=None, names=["tag", "prop"]
        )
        df.index = df.pop("tag")
        # if "less_than_rebound" not in df.tag.values:
        #    df.index = df.pop("tag")
        # df.loc["greater_than-compact3", "prop"] = 1
        for tag in [
            "greater_than-compact1",
            "greater_than-compact2",
            "greater_than-compact3",
            "greater_than-compact4",
        ]:
            df.loc[tag, "prop"] = 0.15

        df.loc["compactatabove.3", "prop"] = 0.33
        df.to_csv(os.path.join(tpl_dir, phi_file), header=False)

    pst.control_data.noptmax = -2
    pst.write(os.path.join(tpl_dir, "pest.pst"), version=2)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    prep_data(True, use_shitty_gsi_data=True, gsi_2_all_layers=False)
