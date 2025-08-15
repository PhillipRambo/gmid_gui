import re
import numpy as np

def extract_all_data(df, vds=0, vgs_interval=None):
    """
    Extract all variable data from the dataframe for a given VDS and optional Vgs interval.
    Automatically detects variables from column names ending with ' Y'.
    Returns a dictionary with keys:
      - 'lengths': 2D array with each row filled with the corresponding length
      - 'vgs': 2D array with each row equal to the Vgs array
      - 'vds': scalar
      - other variables: 2D arrays with shape (num_lengths, num_vgs)
    """
    # --- Step 1: Filter columns ---
    y_cols = [col for col in df.columns if col.endswith(" Y") and f"vds={vds:.2e}" in col]
    x_cols = [col for col in df.columns if col.endswith(" X") and f"vds={vds:.2e}" in col]
    if not x_cols or not y_cols:
        raise ValueError("No matching columns found for the specified VDS.")
    
    x_col = x_cols[0]
    new_df = df[[x_col] + y_cols].rename(columns={x_col: "Vgs"})

    # --- Step 2: Filter Vgs if interval is given ---
    vgs = new_df["Vgs"].to_numpy()
    if vgs_interval is not None:
        mask = (vgs >= vgs_interval[0]) & (vgs <= vgs_interval[1])
        new_df = new_df.loc[mask]
        vgs = new_df["Vgs"].to_numpy()

    # --- Step 3: Extract unique lengths ---
    lengths = np.array(sorted({
        float(m.group(1))
        for col in y_cols
        if (m := re.search(r"length=([0-9.eE+-]+)", col))
    }), dtype=np.float32)

    # --- Step 4: Detect variables ---
    variables = sorted({
        re.search(r'^[^:]+:[^\s(]+', col).group(0)
        for col in y_cols
    })
    
    # --- Step 5: Build data arrays ---
    data_dict = {"vds": vds}
    for var in variables:
        arrays = []
        for l in lengths:
            col = next(c for c in new_df.columns if var in c and f"length={l:.2e}" in c)
            arrays.append(new_df[col].to_numpy(dtype=np.float32))
        data_dict[var] = np.array(arrays, dtype=np.float32)

    # --- Step 6: Create lengths 2D array ---
    example_var = next(k for k in data_dict if k not in ["vds"])
    shape = data_dict[example_var].shape  # (num_lengths, num_vgs)
    data_dict["lengths"] = np.zeros_like(data_dict[example_var], dtype=np.float32)
    for i, l in enumerate(lengths):
        data_dict["lengths"][i, :] = l

    # --- Step 7: Create vgs 2D array ---
    data_dict["vgs"] = np.tile(vgs, (shape[0], 1))

    return data_dict