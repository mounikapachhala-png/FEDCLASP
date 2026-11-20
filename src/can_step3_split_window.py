"""STEP 3 (CAN): TIME-based split per recording (first 70% train, next 10% val, last 20% test),
windows built INSIDE each split (no window crosses a boundary), scaler fit on TRAIN frames only.
Output: data/processed/can_{X,y}_{train,val,test}.npy, can_scaler.joblib, results/split_can.json"""
import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from common import PROC, RES, save_json
from can_io import CLASSES, FEATS, FILES

W, S = 32, 16            # window length, stride (frames)
CUT = (0.70, 0.80)

segs = {"train": [], "val": [], "test": []}
for name in FILES:
    p = PROC / f"can_frames_clean_{name}.parquet"
    if not p.exists():
        print(f"!! {p.name} missing - skipped"); continue
    df = pd.read_parquet(p)
    n = len(df); a, b = int(n * CUT[0]), int(n * CUT[1])
    segs["train"].append(df.iloc[:a]); segs["val"].append(df.iloc[a:b]); segs["test"].append(df.iloc[b:])

scaler = StandardScaler()
for seg in segs["train"]:
    scaler.partial_fit(seg[FEATS].to_numpy(np.float64))   # TRAIN ONLY
joblib.dump(scaler, PROC / "can_scaler.joblib")

summary = {"window": W, "stride": S, "cut": CUT, "classes": CLASSES,
           "label_rule": "window = attack class if ANY frame in it is injected (each recording has one attack type)"}
for split, parts in segs.items():
    Xs, ys = [], []
    for seg in parts:
        if len(seg) < W:
            continue
        X = scaler.transform(seg[FEATS].to_numpy(np.float64)).astype(np.float32)
        y = seg["y"].to_numpy()
        Xs.append(np.ascontiguousarray(sliding_window_view(X, (W, X.shape[1]))[::S, 0]))  # (N, W, F)
        ys.append(sliding_window_view(y, W)[::S].max(axis=1))
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.int64)
    np.save(PROC / f"can_X_{split}.npy", X); np.save(PROC / f"can_y_{split}.npy", y)
    u, c = np.unique(y, return_counts=True)
    summary[split] = {"windows": int(len(y)), "shape": list(X.shape),
                      "class_counts": {CLASSES[i]: int(k) for i, k in zip(u, c)}}
    print(split, summary[split])

save_json(summary, RES / "split_can.json")
