#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
import pandas as pd

STOP_CONSECUTIVE = 5  # 连续满足的批次数


def average_rank_desc(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = v.size
    order = np.argsort(-v, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    pos = 1.0
    while i < n:
        j = i + 1
        vi = v[order[i]]
        while j < n and v[order[j]] == vi:
            j += 1
        k = j - i
        avg = (pos + (pos + k - 1)) / 2.0
        ranks[order[i:j]] = avg
        pos += k
        i = j
    return ranks


def spearman_rho_from_vectors(a: np.ndarray, b: np.ndarray) -> float:
    ra = average_rank_desc(a)
    rb = average_rank_desc(b)
    sa = ra.std()
    sb = rb.std()
    if sa == 0 and sb == 0:
        return 1.0
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 split_rank_spearman.py <app> <test> <comp>")
        sys.exit(1)

    app, test, comp = sys.argv[1], sys.argv[2], sys.argv[3]

    csv_name = f"test_result_{app}_{test}_{comp}.csv"
    csv_path = Path("test_result") / csv_name
    if not csv_path.exists():
        print(f"Input file not found: {csv_path}")
        sys.exit(1)

    # 输出文件路径
    out_dir = Path("order_result")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"order_result_{app}_{test}_{comp}.txt"

    # ---------------- 读取历史结果 ----------------
    history = []
    last_batch = 0
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                try:
                    batch_id = int(parts[0])
                    mean_val = float(parts[1])
                    std_val = float(parts[4])
                    history.append((batch_id, mean_val, std_val))
                    last_batch = batch_id
                except Exception:
                    continue

    # ---------------- 读取 CSV 数据 ----------------
    df = pd.read_csv(csv_path)
    if "kernel" in df.columns:
        mask_valid = ~df["kernel"].astype(str).str.lower().str.startswith("invalid")
    else:
        mask_valid = np.ones(len(df), dtype=bool)

    if "SDC" not in df.columns:
        print("CSV is missing 'SDC' column.")
        sys.exit(1)

    df = df.loc[mask_valid].copy()
    if {"kernel", "inst_line"}.issubset(df.columns):
        df["_inst_id"] = df["kernel"].astype(str) + ":" + df["inst_line"].astype(str)
    else:
        df["_inst_id"] = df.index.astype(str)

    df["SDC"] = pd.to_numeric(df["SDC"], errors="coerce").fillna(0).astype(int)
    total_sdc = int(df["SDC"].sum())
    if total_sdc < 2:
        print(f"Batch {last_batch+1}: insufficient data")
        return

    K = len(df)
    pool = np.repeat(np.arange(K, dtype=np.int32), df["SDC"].to_numpy())

    # ---------------- 分半重复 ----------------
    B = 200
    rhos = np.empty(B, dtype=float)
    for b in range(B):
        perm = np.random.permutation(total_sdc)
        half = total_sdc // 2
        countsA = np.bincount(pool[perm[:half]], minlength=K).astype(float)
        countsB = np.bincount(pool[perm[half:]], minlength=K).astype(float)
        rhos[b] = spearman_rho_from_vectors(countsA, countsB)

    mean_rho = float(np.mean(rhos))
    std_rho = float(np.std(rhos, ddof=1))
    cv_rho = std_rho / mean_rho if mean_rho > 0 else float("inf")
    median_rho = float(np.median(rhos))
    p5_rho = float(np.percentile(rhos, 5))

    batch_idx = last_batch + 1
    print(
        f"Batch {batch_idx}: mean={mean_rho:.6f}, std={std_rho:.6f}, "
        f"cv={cv_rho:.6f}, median={median_rho:.6f}, p5={p5_rho:.6f}"
    )

    # ---------------- 写入结果 ----------------
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(
            f"{batch_idx},{mean_rho:.6f},{median_rho:.6f},"
            f"{p5_rho:.6f},{std_rho:.6f},{cv_rho:.6f}\n"
        )

    # ---------------- 连续阈值判断 ----------------
    history.append((batch_idx, mean_rho, std_rho))
    if len(history) >= STOP_CONSECUTIVE:
        lastN = history[-STOP_CONSECUTIVE:]
        if all(
            m >= 0.9 and (s / m if m > 0 else float("inf")) <= 0.1 for _, m, s in lastN
        ):
            print(
                ">>> Reached 5 consecutive batches meeting threshold (mean>=0.9, CV<=0.1). Stop experiment."
            )
            sys.exit(99)


if __name__ == "__main__":
    main()
