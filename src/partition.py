"""STEP 4: build FL client partitions from the TRAIN split only (same files reused by every method).
Saves an 'owner' array: owner[i] = client id of training sample i.
Usage:
  python src/partition.py --dataset can --all          (all CAN settings)
  python src/partition.py --dataset darknet --all
  python src/partition.py --dataset can --clients 50 --alpha 0.1   (one setting)"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG, PART, PROC, RES, SPLIT_SEED, save_json

SETTINGS = {
    "can":     {"clients": [50, 100, 500, 1000], "alphas": [0.1, 0.3, 1.0], "patho": [(50, 2)]},
    "darknet": {"clients": [10, 20],             "alphas": [0.1, 0.3, 1.0], "patho": [(10, 2)]},
}


def load_y_train(dataset):
    if dataset == "can":
        return np.load(PROC / "can_y_train.npy")
    return np.load(PROC / "darknet_split.npz")["y_train"]


def dirichlet(y, K, alpha, rng, min_size=10, tries=1000):
    C = int(y.max()) + 1
    for _ in range(tries):
        owner = np.empty(len(y), dtype=np.int32)
        for c in range(C):
            idc = np.where(y == c)[0]; rng.shuffle(idc)
            cuts = (np.cumsum(rng.dirichlet(alpha * np.ones(K))) * len(idc)).astype(int)[:-1]
            for k, part in enumerate(np.split(idc, cuts)):
                owner[part] = k
        if np.bincount(owner, minlength=K).min() >= min_size:
            return owner
    raise RuntimeError("Could not satisfy min_size; lower --min_size or clients")


def pathological(y, K, shards_per_client, rng):
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, K * shards_per_client)
    perm = rng.permutation(len(shards))
    owner = np.empty(len(y), dtype=np.int32)
    for j, s in enumerate(perm):
        owner[shards[s]] = j // shards_per_client
    return owner


def save(dataset, tag, owner, y, K):
    C = int(y.max()) + 1
    counts = np.zeros((K, C), dtype=np.int64)
    np.add.at(counts, (owner, y), 1)
    np.save(PART / f"{dataset}_{tag}.npy", owner)
    sizes = counts.sum(1); cls_per_client = (counts > 0).sum(1)
    meta = {"dataset": dataset, "setting": tag, "clients": K, "seed": SPLIT_SEED,
            "client_size_min_median_max": [int(sizes.min()), int(np.median(sizes)), int(sizes.max())],
            "classes_per_client_mean": round(float(cls_per_client.mean()), 2),
            "clients_holding_each_class": (counts > 0).sum(0).tolist()}
    save_json(meta, RES / "partitions" / f"{dataset}_{tag}.json")
    show = counts[: min(K, 50)]
    fig, ax = plt.subplots(figsize=(max(6, show.shape[0] * 0.18), 3))
    im = ax.imshow(np.log1p(show.T), aspect="auto", cmap="viridis")
    ax.set_xlabel("Client id" + (" (first 50)" if K > 50 else "")); ax.set_ylabel("Class")
    ax.set_title(f"{dataset} {tag}: samples per class per client (log1p)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(FIG / f"partition_{dataset}_{tag}.png", dpi=150); plt.close(fig)
    print(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["can", "darknet"], required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--clients", type=int); ap.add_argument("--alpha", type=float)
    ap.add_argument("--patho", type=int, help="classes(shards) per client")
    ap.add_argument("--min_size", type=int, default=10)
    a = ap.parse_args()
    y = load_y_train(a.dataset)
    jobs = []
    if a.all:
        s = SETTINGS[a.dataset]
        jobs += [(K, al, None) for K in s["clients"] for al in s["alphas"]]
        jobs += [(K, None, sh) for K, sh in s["patho"]]
    else:
        jobs.append((a.clients, a.alpha, a.patho))
    for K, al, sh in jobs:
        rng = np.random.default_rng(SPLIT_SEED)
        if sh:
            save(a.dataset, f"K{K}_patho{sh}", pathological(y, K, sh, rng), y, K)
        else:
            save(a.dataset, f"K{K}_dir{al}", dirichlet(y, K, al, rng, a.min_size), y, K)


if __name__ == "__main__":
    main()
