"""Step 0: record software + hardware of THIS PC (needed for the reproducibility table).
Usage:  python src/check_env.py pcA_can      (or pcB_darknet)"""
import json
import platform
import socket
import sys

from common import RES, save_json

tag = sys.argv[1] if len(sys.argv) > 1 else socket.gethostname()
info = {"tag": tag, "python": sys.version.split()[0], "os": platform.platform(),
        "cpu": platform.processor() or platform.machine()}
for mod in ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "pyarrow", "torch", "flwr", "opacus"]:
    try:
        info[mod] = getattr(__import__(mod), "__version__", "?")
    except Exception as e:  # noqa: BLE001
        info[mod] = f"MISSING ({e.__class__.__name__})"
try:
    import psutil
    info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    info["cpu_cores"] = psutil.cpu_count(logical=False)
except ImportError:
    pass
try:
    import torch
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["cuda_version"] = torch.version.cuda
except ImportError:
    pass

save_json(info, RES / f"env_{tag}.json")
print(json.dumps(info, indent=2))
