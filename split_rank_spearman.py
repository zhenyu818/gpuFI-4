#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd


def average_rank_desc(values: np.ndarray) -> np.ndarray:
    """
    Generate average ranks (higher value = higher rank).
    Returns 1-based floating-point ranks.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    order = np.argsort(-v, kind="mergesort")  # stable sort, descending
    ranks = np.empty(n, dtype=float)

    i = 0
    pos = 1.0  # 1-based
    while i < n:
        j = i + 1
        vi = v[order[i]]
        while j < n and v[order[j]] == vi:
            j += 1
        # (i..j-1) are ties, assign average rank
        k = j - i
        avg = (pos + (pos + k - 1)) / 2.0
        ranks[order[i:j]] = avg
        pos += k
        i = j
    return ranks


def spearman_rho_from_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """
    Convert two vectors to average ranks (descending),
    then compute Pearson correlation between ranks.
    """
    ra = average_rank_desc(a)
    rb = average_rank_desc(b)

    sa = ra.std()
    sb = rb.std()
    if sa == 0 and sb == 0:
        return 1.0  # both constant: perfect match
    if sa == 0 or sb == 0:
        return 0.0  # one constant, one not: undefined, set to 0
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 split_rank_spearman.py <app> <test> <comp> <i>")
        sys.exit(1)

    app, test, comp, i_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    try:
        batch_idx = int(i_str)
    except Exception:
        print("The 4th argument i must be an integer.")
        sys.exit(1)

    # Input file is now located under test_result/
    csv_name = f"test_result_{app}_{test}_{comp}.csv"
    csv_path = Path("test_result") / csv_name
    if not csv_path.exists():
        print(f"Input file not found: {csv_path}")
        sys.exit(1)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Ignore invalid rows
    if 'kernel' in df.columns:
        mask_valid = ~df['kernel'].astype(str).str.lower().str.startswith('invalid')
    else:
        mask_valid = np.ones(len(df), dtype=bool)

    # Ensure SDC column exists
    if 'SDC' not in df.columns:
        print("CSV is missing 'SDC' column.")
        sys.exit(1)

    df = df.loc[mask_valid].copy()
    # Unique instruction ID
    if {'kernel', 'inst_line'}.issubset(df.columns):
        df['_inst_id'] = df['kernel'].astype(str) + ":" + df['inst_line'].astype(str)
    else:
        df['_inst_id'] = df.index.astype(str)

    df['SDC'] = pd.to_numeric(df['SDC'], errors='coerce').fillna(0).astype(int)

    total_sdc = int(df['SDC'].sum())
    if total_sdc < 2:
        # Degenerate case
        mean_rho = median_rho = p5_rho = std_rho = float('nan')
        line = f"Batch {batch_idx}: mean={mean_rho}, median={median_rho}, p5={p5_rho}, std={std_rho}"
        print(line)
        out_dir = Path("order_result")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"order_result_{app}_{test}_{comp}.txt"
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"{batch_idx},{mean_rho},{median_rho},{p5_rho},{std_rho}\n")
        sys.exit(0)

    inst_ids = df['_inst_id'].tolist()
    sdc_counts = df['SDC'].to_numpy()
    K = len(inst_ids)

    pool = np.repeat(np.arange(K, dtype=np.int32), sdc_counts)

    B = 200  # number of repetitions
    rhos = np.empty(B, dtype=float)

    for b in range(B):
        perm = np.random.permutation(total_sdc)
        half = total_sdc // 2
        idxA = perm[:half]
        idxB = perm[half:]

        partA = pool[idxA]
        partB = pool[idxB]

        countsA = np.bincount(partA, minlength=K).astype(float)
        countsB = np.bincount(partB, minlength=K).astype(float)

        rhos[b] = spearman_rho_from_vectors(countsA, countsB)

    mean_rho = float(np.mean(rhos))
    median_rho = float(np.median(rhos))
    p5_rho = float(np.percentile(rhos, 5))
    std_rho = float(np.std(rhos, ddof=1))

    # Print results
    print(f"Batch {batch_idx}: mean={mean_rho:.6f}, median={median_rho:.6f}, p5={p5_rho:.6f}, std={std_rho:.6f}")

    # Append to file
    out_dir = Path("order_result")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"order_result_{app}_{test}_{comp}.txt"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"{batch_idx},{mean_rho:.6f},{median_rho:.6f},{p5_rho:.6f},{std_rho:.6f}\n")


if __name__ == "__main__":
    main()
