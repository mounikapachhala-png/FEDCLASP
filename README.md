# FedCLASP — Steps 0–4 (both PCs)

PC-A = CAN-bus (HCRL Car-Hacking) — use the PC with the GPU (windows are larger).
PC-B = CIC-Darknet2020 — CPU is fine for Steps 1–4.
Same code, same versions, same seeds on both PCs. Only the data folder differs.

## Step 0 — Set up VS Code (do this on BOTH PCs, identically)

0.1 Install Python 3.11 (same minor version on both PCs), Git, and VS Code.
    In VS Code install extensions: "Python" and "Pylance" (Microsoft).
0.2 One person creates a PRIVATE GitHub repo, unzips this folder into it, commits, pushes.
    The other PC: `git clone <repo-url>`. Data and big arrays are git-ignored on purpose.
0.3 VS Code → File → Open Folder → the repo. Terminal → New Terminal, then:
        python -m venv .venv
        .venv\Scripts\activate          (Windows)      |  source .venv/bin/activate  (Linux/Mac)
    Ctrl+Shift+P → "Python: Select Interpreter" → pick .venv
0.4 Install PyTorch first (pick your CUDA at pytorch.org), e.g.
        pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121   (GPU PC)
        pip install torch==2.3.1                                                       (CPU PC)
    then:  pip install -r requirements.txt
0.5 Record the machine (goes into the paper's reproducibility table):
        python src/check_env.py pcA_can        (on PC-A)
        python src/check_env.py pcB_darknet    (on PC-B)
        pip freeze > results/freeze_pcA_can.txt   (or pcB_darknet)

All commands below are run from the repo root with .venv active.

## PC-A — CAN-bus

Put the files in data/raw/can/:  DoS_dataset.csv, Fuzzy_dataset.csv, gear_dataset.csv, RPM_dataset.csv
(If your CAN dataset is NOT HCRL Car-Hacking, stop and adapt src/can_io.py first.)

    python src/can_step1_inspect.py        # Step 1 -> results/table1_can.json  (parsing takes minutes)
    python src/can_step2_clean.py          # Step 2 -> results/cleaning_log_can.json
    python src/can_step3_split_window.py   # Step 3 -> results/split_can.json (time split, windows 32/16)
    python src/partition.py --dataset can --all   # Step 4 -> data/partitions/, results/figures/

CHECK after Step 3: open results/split_can.json. "Normal" must be the LARGEST window class.
If Normal windows are tiny, attacks are spread through the file and the "any attack frame"
rule is mislabelling — report it before going on.

## PC-B — Darknet

Put Darknet.CSV in data/raw/darknet/. Decide the label in src/darknet_io.py
(LABEL_MODE = "traffic" 4-class, default; or "app" 8-class) and write the choice in your log.

    python src/darknet_step1_inspect.py    # Step 1 -> results/table1_darknet.json
    python src/darknet_step2_clean.py      # Step 2 -> results/cleaning_log_darknet.json
    python src/darknet_step3_split.py      # Step 3 -> results/split_darknet.json
    python src/partition.py --dataset darknet --all   # Step 4

## After each step (both PCs)
    git add results src && git commit -m "PC-A step N" && git pull --rebase && git push
Only small JSON/code goes to Git. Never edit SPLIT_SEED after Step 3.

## Done criteria (from the guide)
Step 1: table1_*.json · Step 2: cleaning_log_*.json · Step 3: split_*.json + scaler
Step 4: partition files for every alpha/client count + partition_*.png heatmaps
