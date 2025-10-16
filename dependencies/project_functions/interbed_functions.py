import numpy as np

def interbed_thicknesses(lay_df, lith, quantiles, use_delay):
    for aq in lay_df.columns:
        aq_lith = lith.loc[lith.Aquifer == aq]
        clay_beds = []
        bb = 0.0
        for idx, (b, on_lith) in enumerate(zip(aq_lith["Thick_ft"], aq_lith["Description"])):
            if "clay" not in on_lith:
                if bb > 0:
                    clay_beds.append(float(bb))
                bb = 0.0
                continue
            bb += b
            if  idx + 1 == aq_lith.shape[0]:
                clay_beds.append(float(bb))
        clay_beds = np.sort(clay_beds)
        if len(clay_beds) == 0:
            quantiles = None
        if quantiles is None:
            clay_beds = [clay_beds]
        
        else:
            quantile_values = np.quantile(clay_beds, quantiles)
            split_indices = np.searchsorted(clay_beds, quantile_values)
            clay_beds = np.split(clay_beds, split_indices)
            clay_beds = [arr for arr in clay_beds if arr.shape[0] > 0]
        for idx, arr in enumerate(clay_beds):
            lay_df.loc[f"clay_thickness_{idx}", aq] = arr.sum()
            bequiv = np.sqrt((arr**2).sum() / float(arr.shape[0]))
            rnb = lay_df.loc[f"clay_thickness_{idx}", aq] / bequiv
            if use_delay:
                thick_frac = bequiv
            else:
                thick_frac = arr.sum()

            lay_df.loc[f"thick_frac_{idx}", aq] = thick_frac
            lay_df.loc[f"rnb_{idx}", aq] = rnb
            lay_df.loc[f"clay_lith_{idx}", aq] = len(aq_lith)
            lay_df.loc[f"clay_thickness_squared_{idx}", aq] = (arr**2).sum()

    return lay_df
