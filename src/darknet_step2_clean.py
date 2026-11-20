"""STEP 2 (Darknet): remove identity/leaky columns, NaN/Inf, constants, duplicates.
Output: data/processed/darknet_clean.parquet + results/cleaning_log_darknet.json"""
import numpy as np
import pandas as pd

from common import PROC, RES, save_json
from darknet_io import ID_COLS, LABEL_MODE, load_raw

df = load_raw()
log = {"label_mode": LABEL_MODE, "rows_raw": len(df)}
LABELS = ["y_traffic", "y_app"]

drop_id = [c for c in ID_COLS if c in df.columns]
df = df.drop(columns=drop_id)
log["dropped_identity_columns"] = {c: "identity/endpoint leakage (lets model memorise hosts)" for c in drop_id}

feat = [c for c in df.columns if c not in LABELS]
df[feat] = df[feat].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
n = len(df); df = df.dropna(); log["rows_dropped_nan_or_inf"] = n - len(df)

unk = df[LABELS].apply(lambda s: s.str.startswith("UNKNOWN")).any(axis=1)
log["rows_dropped_unknown_label"] = int(unk.sum()); df = df[~unk]

const = [c for c in feat if df[c].nunique() <= 1]
df = df.drop(columns=const); feat = [c for c in feat if c not in const]
log["dropped_constant_columns"] = const

n = len(df); df = df.drop_duplicates(); log["rows_dropped_exact_duplicates"] = n - len(df)
# Same features but different label = contradictory sample -> remove all copies
amb = df.duplicated(subset=feat, keep=False)
log["rows_dropped_conflicting_duplicates"] = int(amb.sum()); df = df[~amb]

target = "y_traffic" if LABEL_MODE == "traffic" else "y_app"
classes = sorted(df[target].unique())
label_map = {c: i for i, c in enumerate(classes)}
df["y"] = df[target].map(label_map).astype(np.int64)
df[feat] = df[feat].astype(np.float32)

log.update(rows_final=len(df), n_features=len(feat), features=feat, label_map=label_map,
           class_counts=df[target].value_counts().to_dict())
df[feat + ["y", "y_traffic", "y_app"]].reset_index(drop=True).to_parquet(PROC / "darknet_clean.parquet", index=False)
save_json(log, RES / "cleaning_log_darknet.json")
for k, v in log.items():
    if k != "features":
        print(f"{k:38s}: {v}")
