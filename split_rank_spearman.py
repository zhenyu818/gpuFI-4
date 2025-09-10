#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
import pandas as pd

THRESH_MEDIAN = 0.95
THRESH_P5 = 0.94
STOP_CONSECUTIVE = 5


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
    if len(sys.argv) < 5:
        print("Usage: python3 split_rank_spearman.py <app> <test> <comp> <i>")
        sys.exit(1)

    app, test, comp, i_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    batch_idx = int(i_str)

    csv_name = f"test_result_{app}_{test}_{comp}.csv"
    csv_path = Path("test_result") / csv_name
    if not csv_path.exists():
        print(f"Input file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if 'kernel' in df.columns:
        mask_valid = ~df['kernel'].astype(str).str.lower().str.startswith('invalid')
    else:
        mask_valid = np.ones(len(df), dtype=bool)

    if 'SDC' not in df.columns:
        print("CSV is missing 'SDC' column.")
        sys.exit(1)

    df = df.loc[mask_valid].copy()
    if {'kernel', 'inst_line'}.issubset(df.columns):
        df['_inst_id'] = df['kernel'].astype(str) + ":" + df['inst_line'].astype(str)
    else:
        df['_inst_id'] = df.index.astype(str)

    df['SDC'] = pd.to_numeric(df['SDC'], errors='coerce').fillna(0).astype(int)
    total_sdc = int(df['SDC'].sum())
    if total_sdc < 2:
        print(f"Batch {batch_idx}: insufficient data")
        return

    K = len(df)
    pool = np.repeat(np.arange(K, dtype=np.int32), df['SDC'].to_numpy())

    # ---------- 200 次分半 ----------
    B = 200
    rhos = np.empty(B, dtype=float)
    for b in range(B):
        perm = np.random.permutation(total_sdc)
        half = total_sdc // 2
        countsA = np.bincount(pool[perm[:half]], minlength=K).astype(float)
        countsB = np.bincount(pool[perm[half:]], minlength=K).astype(float)
        rhos[b] = spearman_rho_from_vectors(countsA, countsB)

    mean_rho = float(np.mean(rhos))
    median_rho = float(np.median(rhos))
    p5_rho = float(np.percentile(rhos, 5))
    std_rho = float(np.std(rhos, ddof=1))

    print(f"Batch {batch_idx}: mean={mean_rho:.6f}, median={median_rho:.6f}, "
          f"p5={p5_rho:.6f}, std={std_rho:.6f}")

    # ---------- 写结果 ----------
    out_dir = Path("order_result")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"order_result_{app}_{test}_{comp}.txt"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"{batch_idx},{mean_rho:.6f},{median_rho:.6f},"
                f"{p5_rho:.6f},{std_rho:.6f}\n")

    # ---------- 判断是否连续 STOP_CONSECUTIVE 次 ----------
    lines = open(out_path, encoding="utf-8").read().strip().splitlines()
    lastN = lines[-STOP_CONSECUTIVE:] if len(lines) >= STOP_CONSECUTIVE else []
    consec_ok = False
    if len(lastN) == STOP_CONSECUTIVE:
        consec_ok = True
        for line in lastN:
            parts = line.split(",")
            try:
                med = float(parts[2])
                p5 = float(parts[3])
            except Exception:
                consec_ok = False
                break
            if not (med >= THRESH_MEDIAN and p5 >= THRESH_P5):
                consec_ok = False
                break
    if consec_ok:
        print(">>> Reached 5 consecutive batches meeting threshold. Stop experiment.")
        sys.exit(99)  # 特殊退出码，bash 检查这个码来 break


if __name__ == "__main__":
    main()
