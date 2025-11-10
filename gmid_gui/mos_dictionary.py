import re
import numpy as np



def extract_all_data(df, vds=0, vgs_interval=None, vds_tol=1e-3):
    """
    Extracts all variable data from the dataframe for a given VDS
    and optional VGS interval.
    Works even if the column prefix is simple like 'N0:1'.
    """
    # Step 1: Select columns matching the requested VDS
    y_cols, x_cols = [], []
    for col in df.columns:
        if col.endswith((" Y", " X")):
            m = re.search(r"vds=([0-9.eE+-]+)", col)
            if not m:
                continue
            vds_val = float(m.group(1))
            if abs(vds_val - vds) < vds_tol:
                if col.endswith(" Y"):
                    y_cols.append(col)
                else:
                    x_cols.append(col)

    if not x_cols or not y_cols:
        raise ValueError(f"No matching columns found for VDS ≈ {vds}")

    # Step 2: Identify unique lengths
    lengths = np.array(sorted({
        float(m.group(1))
        for col in y_cols
        if (m := re.search(r"length=([0-9.eE+-]+)", col))
    }), dtype=np.float32)

    # Step 3: Detect variable prefixes (handle nodes like N0:1)
    variables = set()
    for col in y_cols:
        m = re.search(r'^[^()]+', col)
        if m:
            prefix = m.group(0).strip()
            if prefix:
                variables.add(prefix)
    variables = sorted(variables)

    # Step 4: Reference X-column for Vgs
    ref_x = x_cols[0]
    vgs = df[ref_x].to_numpy(dtype=np.float32)
    if vgs_interval is not None:
        mask = (vgs >= vgs_interval[0]) & (vgs <= vgs_interval[1])
        vgs = vgs[mask]
    else:
        mask = slice(None)

    # Step 5: Build arrays per variable and length
    data_dict = {"vds": np.float32(vds)}
    for var in variables:
        arrays = []
        for L in lengths:
            pattern = f"{var} (length={L:.2e},vds="
            y_col = next((c for c in y_cols if c.startswith(pattern)), None)
            if y_col is None:
                continue
            y_data = df[y_col].to_numpy(dtype=np.float32)[mask]
            arrays.append(y_data)
        if arrays:
            data_dict[var] = np.array(arrays, dtype=np.float32)

    # Step 6: Build helper arrays
    example_var = next((k for k in data_dict if k not in ["vds"]), None)
    if example_var:
        shape = data_dict[example_var].shape
        data_dict["lengths"] = np.tile(lengths[:, None], (1, shape[1]))
        data_dict["vgs"] = np.tile(vgs[None, :], (shape[0], 1))

    return data_dict