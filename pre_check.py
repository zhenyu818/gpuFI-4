#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse


def _check_consecutive_5(rows, thr=0.99):
    """
    检查最近 5 次 Sp_SDC >= thr
    rows: 从 result_info 文件读取的所有行（包含表头）
    """
    if not rows or len(rows) <= 1:
        return False

    # 跳过表头
    data_rows = rows[1:]

    if len(data_rows) < 5:
        return False

    last5 = data_rows[-5:]

    def geq(x):
        try:
            return float(x) >= thr
        except Exception:
            return False

    # r[2] 对应 Sp_SDC
    return all(geq(r[2]) for r in last5)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-check exit condition before main program"
    )
    parser.add_argument("--app", "-a", required=True, help="Application name")
    parser.add_argument("--test", "-t", required=True, help="Test identifier", type=str)
    parser.add_argument(
        "--component", "-c", required=False, help="Component set", type=int, default=""
    )
    parser.add_argument(
        "--inject_count",
        "-i",
        required=False,
        help="Inject bit flip count",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    info_dir = "result_info"
    info_path = os.path.join(info_dir, f"result_info_{args.app}_{args.test}_{args.component}_{args.inject_count}.csv")

    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if _check_consecutive_5(rows, thr=0.99):
            print("Sp_SDC have been >= 0.99 for 5 consecutive runs (history check).")
            sys.exit(99)

    # 如果没有满足条件，正常返回 0，主程序可以继续执行
    sys.exit(0)


if __name__ == "__main__":
    main()
