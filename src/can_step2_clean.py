"""STEP 2 (CAN): sort, de-duplicate, add timing features (past-only, so no future leakage).
Output: data/processed/can_frames_clean_<attack>.parquet + results/cleaning_log_can.json"""
import numpy as np
import pandas as pd

from common import PROC, RES, save_json
from can_io import FEATS, FILES

log = {}
for name in FILES:
    p = PROC / f"can_frames_{name}.parquet"
    if not p.exists():
        print(f"!! run can_step1_inspect.py first ({p.name} missing)"); continue
    df = pd.read_parquet(p)
    n0 = len(df)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df = df.drop_duplicates(subset=["ts", "can_id", "dlc"] + [f"b{i}" for i in range(8)]).reset_index(drop=True)
    # timing features use only PAST frames (diff) -> causal, deployable on an ECU gateway
    dt_prev = df["ts"].diff().fillna(0.0).clip(lower=0)
    dt_same = df.groupby("can_id")["ts"].diff().fillna(0.0).clip(lower=0)
    df["dt_prev"] = np.log1p(dt_prev * 1e3)       # ms, log-compressed
    df["dt_same_id"] = np.log1p(dt_same * 1e3)
    df[FEATS + ["ts", "y"]].to_parquet(PROC / f"can_frames_clean_{name}.parquet", index=False)
    log[name] = {"rows_in": n0, "rows_dropped_duplicates": n0 - len(df), "rows_out": len(df)}
    print(name, log[name])

log["features"] = FEATS
log["removed_columns"] = {"flag": "raw label source (converted to y)",
                          "ts (as a feature)": "absolute time = recording identity leakage; kept only for ordering"}
save_json(log, RES / "cleaning_log_can.json")
