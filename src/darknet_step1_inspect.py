"""STEP 1 (Darknet): look inside the data -> results/table1_darknet.json"""
import numpy as np
import pandas as pd

from common import RES, imbalance_ratio, save_json
from darknet_io import ID_COLS, find_csv, load_raw

df = load_raw()
feat = [c for c in df.columns if c not in ("y_traffic", "y_app") and c not in ID_COLS]
num = df[feat].apply(pd.to_numeric, errors="coerce")

info = {
    "file": find_csv().name,
    "rows": len(df),
    "columns_total": df.shape[1],
    "feature_columns_after_id_drop": len(feat),
    "identity_columns_present": [c for c in ID_COLS if c in df.columns],
    "rows_with_nan": int(num.isna().any(axis=1).sum()),
    "rows_with_inf": int(np.isinf(num.to_numpy(dtype=float, na_value=0)).any(axis=1).sum()),
    "exact_duplicate_rows": int(df.duplicated().sum()),
    "constant_columns": [c for c in feat if num[c].nunique(dropna=True) <= 1],
}
for col in ("y_traffic", "y_app"):
    vc = df[col].value_counts().to_dict()
    info[f"{col}_counts"] = vc
    info[f"{col}_imbalance_ratio"] = imbalance_ratio(vc)
    unknown = [k for k in vc if str(k).startswith("UNKNOWN")]
    if unknown:
        print(f"!! Unmapped labels in {col}: {unknown}  -> add them to the maps in darknet_io.py")

save_json(info, RES / "table1_darknet.json")
for k, v in info.items():
    print(f"{k:35s}: {v}")
print("\nColumns:", list(df.columns))
