"""STEP 1 (CAN): parse every attack file, summarise -> results/table1_can.json
Also caches parsed frames to data/processed/can_frames_<attack>.parquet (parsing is slow)."""
from common import PROC, RES, imbalance_ratio, save_json
from can_io import CAN_DIR, CLASSES, FILES, parse_hcrl

summary, total = {}, {c: 0 for c in CLASSES}
for name, fn in FILES.items():
    path = CAN_DIR / fn
    if not path.exists():
        print(f"!! missing {path} - skipped"); continue
    print(f"parsing {fn} ...")
    df, info = parse_hcrl(path, name)
    df.to_parquet(PROC / f"can_frames_{name}.parquet", index=False)
    att = int((df["flag"] == "T").sum())
    info.update(frames=len(df), attack_frames=att, normal_frames=len(df) - att,
                attack_fraction=round(att / len(df), 4),
                unique_can_ids=int(df["can_id"].nunique()),
                duration_s=round(float(df["ts"].max() - df["ts"].min()), 1),
                timestamps_sorted=bool(df["ts"].is_monotonic_increasing),
                dlc_distribution=df["dlc"].value_counts().sort_index().to_dict(),
                injected_can_ids=sorted(df.loc[df["flag"] == "T", "can_id"].unique().tolist())[:20])
    summary[name] = info
    total["Normal"] += len(df) - att; total[name] += att
    print(info)

summary["frame_level_class_counts"] = total
summary["frame_level_imbalance_ratio"] = imbalance_ratio(total)
summary["note"] = "Single vehicle, one recording per attack -> no natural clients; window-level counts come in Step 3"
save_json(summary, RES / "table1_can.json")
print("\nTotal frames per class:", total, "| imbalance:", summary["frame_level_imbalance_ratio"])
