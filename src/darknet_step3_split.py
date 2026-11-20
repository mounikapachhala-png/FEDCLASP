"""STEP 3 (Darknet): stratified 70/10/20 split FIRST, then scaler fitted on TRAIN ONLY.
Output: data/processed/darknet_split.npz, darknet_scaler.joblib, results/split_darknet.json"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import PROC, RES, SPLIT_SEED, save_json

df = pd.read_parquet(PROC / "darknet_clean.parquet")
feat = [c for c in df.columns if c not in ("y", "y_traffic", "y_app")]
X = df[feat].to_numpy(np.float32); y = df["y"].to_numpy(np.int64)

idx = np.arange(len(y))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SPLIT_SEED)
va, te = train_test_split(tmp, test_size=2 / 3, stratify=y[tmp], random_state=SPLIT_SEED)
assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))

scaler = StandardScaler().fit(X[tr])           # TRAIN ONLY
f = lambda a: np.clip(scaler.transform(a), -10, 10).astype(np.float32)  # clip heavy tails
np.savez_compressed(PROC / "darknet_split.npz",
                    X_train=f(X[tr]), y_train=y[tr], X_val=f(X[va]), y_val=y[va],
                    X_test=f(X[te]), y_test=y[te], features=np.array(feat))
joblib.dump(scaler, PROC / "darknet_scaler.joblib")

summary = {"seed": SPLIT_SEED, "scaler": "StandardScaler fit on train, clip [-10,10]",
           **{s: {"n": int(len(i)), "class_counts": dict(zip(*map(lambda a: a.tolist(), np.unique(y[i], return_counts=True))))}
              for s, i in (("train", tr), ("val", va), ("test", te))}}
save_json(summary, RES / "split_darknet.json")
print(summary)
