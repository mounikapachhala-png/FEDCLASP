"""Loader for CIC-Darknet2020 (Darknet.CSV). Normalises the known messy label spellings."""
import pandas as pd

from common import RAW

DARKNET_DIR = RAW / "darknet"

# DECISION (write it in the paper): "traffic" = 4-class Tor / Non-Tor / VPN / Non-VPN
#                                   "app"     = 8-class application label
LABEL_MODE = "traffic"

ID_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]
TRAFFIC_MAP = {"tor": "Tor", "nontor": "Non-Tor", "vpn": "VPN", "nonvpn": "Non-VPN"}
APP_MAP = {"audiostreaming": "Audio-Streaming", "browsing": "Browsing", "chat": "Chat",
           "email": "Email", "filetransfer": "File-Transfer", "p2p": "P2P",
           "videostreaming": "Video-Streaming", "voip": "VOIP"}


def _key(v) -> str:
    return "".join(ch for ch in str(v).lower() if ch.isalnum())


def find_csv():
    files = sorted(p for p in DARKNET_DIR.iterdir() if p.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"Put Darknet.CSV inside {DARKNET_DIR}")
    return files[0]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(find_csv(), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "Label" not in df.columns or "Label.1" not in df.columns:
        raise KeyError(f"Expected 'Label' and 'Label.1' columns, got: {list(df.columns)[-5:]}")
    df["y_traffic"] = df["Label"].map(lambda v: TRAFFIC_MAP.get(_key(v), f"UNKNOWN:{v}"))
    df["y_app"] = df["Label.1"].map(lambda v: APP_MAP.get(_key(v), f"UNKNOWN:{v}"))
    return df.drop(columns=["Label", "Label.1"])
