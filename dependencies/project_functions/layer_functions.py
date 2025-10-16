import numpy as np
def layer_thicknesses(lay_df, lith, top):
    botm = []
    last = top
    for aq in lay_df.columns:
        t = lith.loc[lith.Aquifer == aq, "Thick_ft"].sum()
        b = last - lith.loc[lith.Aquifer == aq, "Bot"].max()
        lay_df.loc["tot_thick", aq] = t
        lay_df.loc["thk", aq] = t
        botm.append(b)

        last = b

    lay_df.loc["botm", :] = np.array(botm)
    lay_df.loc["top", :] = top
    return lay_df