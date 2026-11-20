"""Parser for the HCRL Car-Hacking dataset (DoS / Fuzzy / gear / RPM CSVs).
Row format: Timestamp, CAN ID(hex), DLC, DATA[0..DLC-1](hex), Flag(T=injected, R=normal).
Rows have a VARIABLE number of fields because DLC varies -> parsed manually.
If your CAN dataset is different, change FILES/CLASSES or send 5 raw lines to adapt."""
import numpy as np
import pandas as pd

from common import RAW

CAN_DIR = RAW / "can"
FILES = {"DoS": "DoS_dataset.csv", "Fuzzy": "Fuzzy_dataset.csv",
         "Gear": "gear_dataset.csv", "RPM": "RPM_dataset.csv"}
CLASSES = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]   # list index = integer label
FEATS = ["can_id", "dlc"] + [f"b{i}" for i in range(8)] + ["dt_prev", "dt_same_id"]


def _hex(s: pd.Series) -> np.ndarray:
    def f(h):
        try:
            return int(str(h).strip(), 16)
        except ValueError:
            return -1
    return s.map(f).to_numpy(dtype=np.int64)


def parse_hcrl(path, attack_name):
    raw = pd.read_csv(path, header=None, names=list(range(12)), dtype=str, skip_blank_lines=True)
    n_raw = len(raw)
    raw[2] = pd.to_numeric(raw[2], errors="coerce")
    raw = raw[raw[2].between(0, 8)].reset_index(drop=True)
    n_bad_dlc = n_raw - len(raw)

    dlc = raw[2].astype(int).to_numpy()
    ts = pd.to_numeric(raw[0], errors="coerce").to_numpy()
    can_id = _hex(raw[1])
    data = np.zeros((len(raw), 8), dtype=np.int64)
    flag = np.full(len(raw), "", dtype=object)
    for d in np.unique(dlc):
        m = dlc == d
        for i in range(d):
            data[m, i] = _hex(raw.loc[m, 3 + i])
        flag[m] = raw.loc[m, 3 + d].astype(str).str.strip().str.upper().to_numpy()

    bad = np.isnan(ts) | (can_id < 0) | (data < 0).any(axis=1) | ~np.isin(flag, ["T", "R"])
    df = pd.DataFrame({"ts": ts, "can_id": can_id, "dlc": dlc})
    for i in range(8):
        df[f"b{i}"] = data[:, i]
    df["flag"] = flag
    df = df[~bad].reset_index(drop=True)
    df["y"] = np.where(df["flag"] == "T", CLASSES.index(attack_name), 0).astype(np.int64)
    return df, {"rows_raw": n_raw, "rows_bad_parse": int(bad.sum()) + n_bad_dlc}
