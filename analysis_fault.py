#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import sys
from collections import defaultdict, Counter
from copy import deepcopy


def normalize_result(s: str) -> str:
    """标准化结果为 Masked / SDC / DUE / Others"""
    x = s.strip().lower()
    if "sdc" in x:
        return "SDC"
    if "due" in x:
        return "DUE"
    if "masked" in x:
        return "Masked"
    return "Others"


def parse_log(log_path: str):
    """逐条解析日志"""
    re_effects_start = re.compile(r"^\[Run\s+(\d+)\]\s+Effects from\s+(?:.+/)?(tmp\.out\d+):\s*$")
    re_writer = re.compile(
        r"^\[(?P<src>[A-Za-z0-9_]+)_FI_WRITER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    re_reader = re.compile(
        r"^\[(?P<src>[A-Za-z0-9_]+)_FI_READER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    re_result = re.compile(r"^\[Run\s+(\d+)\]\s+(tmp\.out\d+):\s*(.*?)\s*$")
    # [INJ_PARAMS] [Run <id>] tmp.outN key-vals
    re_params = re.compile(r"^\[INJ_PARAMS\]\s+\[Run\s+(\d+)\]\s+(tmp\.out\d+)\s+(.*)$")

    latest_effects_by_pair = {}
    params_by_pair = {}
    cur_key = None
    cur_writers, cur_readers = [], []

    occ_counter = defaultdict(int)
    effects_occ, results_occ = {}, {}

    def build_recs():
        if cur_writers:
            return deepcopy(cur_writers)
        if cur_readers:
            return deepcopy(cur_readers)
        return [{
            "kernel": "invalid_summary",
            "inst_line": -1,
            "inst_text": "",
            "src": "invalid",
        }]

    def flush_current_effects():
        nonlocal cur_key, cur_writers, cur_readers
        if cur_key is not None:
            latest_effects_by_pair[cur_key] = build_recs()
            cur_key = None
            cur_writers, cur_readers = [], []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")

                m = re_effects_start.match(line)
                if m:
                    flush_current_effects()
                    run_id = int(m.group(1))
                    name = m.group(2)
                    cur_key = (run_id, name)
                    cur_writers, cur_readers = [], []
                    continue

                if cur_key is not None:
                    m = re_writer.match(line)
                    if m:
                        cur_writers = [{
                            "kernel": m.group(2),
                            "inst_line": int(m.group(4)),
                            "inst_text": m.group(5).strip(),
                            "src": m.group("src"),
                        }]
                        continue
                    m = re_reader.match(line)
                    if m and not cur_writers:
                        cur_readers.append({
                            "kernel": m.group(2),
                            "inst_line": int(m.group(4)),
                            "inst_text": m.group(5).strip(),
                            "src": m.group("src"),
                        })
                        continue

                m = re_params.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    params_by_pair[(run_id, name)] = m.group(3).strip()
                    continue

                m = re_result.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    res = normalize_result(m.group(3))
                    pair = (run_id, name)

                    occ_counter[pair] += 1
                    idx = occ_counter[pair]
                    inj_key = (run_id, name, idx)

                    if cur_key == pair:
                        recs = build_recs()
                        latest_effects_by_pair[pair] = deepcopy(recs)
                    else:
                        recs = latest_effects_by_pair.get(pair, [{
                            "kernel": "invalid_summary",
                            "inst_line": -1,
                            "inst_text": "",
                            "src": "invalid",
                        }])

                    effects_occ[inj_key] = deepcopy(recs)
                    results_occ[inj_key] = res
                    continue

        flush_current_effects()

    except FileNotFoundError:
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    return effects_occ, results_occ, params_by_pair


def write_csv(app: str, test: str, effects_occ, results_occ):
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_result_{app}_{test}.csv")

    inst_counts = defaultdict(lambda: defaultdict(lambda: {"Masked": 0, "SDC": 0, "DUE": 0, "Others": 0}))
    all_srcs = set()

    # === 合并旧 CSV ===
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel = row["kernel"]
                inst_line = -1 if row["inst_line"] == "" else int(row["inst_line"])
                inst_text = row["inst_text"]
                key = (kernel, inst_line, inst_text)
                for col, val in row.items():
                    if col in ("kernel", "inst_line", "inst_text"):
                        continue
                    if col.endswith(("_Masked", "_SDC", "_DUE", "_Others")):
                        src, cat = col.rsplit("_", 1)
                        inst_counts[key][src][cat] += int(val)
                        all_srcs.add(src)

    # === 本次注入结果 ===
    for inj_key, recs in effects_occ.items():
        res_cat = results_occ.get(inj_key, "Others")
        for rec in recs:
            kernel = rec.get("kernel") or "unknown"
            inst_line = rec.get("inst_line") if rec.get("inst_line") is not None else -1
            inst_text = rec.get("inst_text") or "unknown"
            src = rec.get("src", "unknown")
            key = (kernel, inst_line, inst_text)
            inst_counts[key][src][res_cat] += 1
            all_srcs.add(src)

    # === 写回 CSV ===
    src_columns = []
    for src in sorted(all_srcs):
        src_columns += [f"{src}_Masked", f"{src}_SDC", f"{src}_DUE", f"{src}_Others"]

    fieldnames = ["kernel", "inst_line", "inst_text"] + src_columns + [
        "Masked", "SDC", "DUE", "Others", "tot_inj"
    ]

    total_masked = total_sdc = total_due = total_others = 0

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (kernel, inst_line, inst_text) in sorted(inst_counts.keys(), key=lambda k: (k[0], k[1], k[2])):
            src_map = inst_counts[(kernel, inst_line, inst_text)]
            row = {
                "kernel": kernel,
                "inst_line": "" if inst_line < 0 else inst_line,
                "inst_text": inst_text,
            }
            tot_m = tot_s = tot_d = tot_o = 0
            for src in all_srcs:
                m = src_map.get(src, {}).get("Masked", 0)
                s = src_map.get(src, {}).get("SDC", 0)
                d = src_map.get(src, {}).get("DUE", 0)
                o = src_map.get(src, {}).get("Others", 0)
                row[f"{src}_Masked"] = m
                row[f"{src}_SDC"] = s
                row[f"{src}_DUE"] = d
                row[f"{src}_Others"] = o
                tot_m += m
                tot_s += s
                tot_d += d
                tot_o += o

            row["Masked"] = tot_m
            row["SDC"] = tot_s
            row["DUE"] = tot_d
            row["Others"] = tot_o
            row["tot_inj"] = tot_m + tot_s + tot_d + tot_o

            total_masked += tot_m
            total_sdc += tot_s
            total_due += tot_d
            total_others += tot_o

            writer.writerow(row)

    total_inj = total_masked + total_sdc + total_due + total_others
    return out_path, total_masked, total_sdc, total_due, total_others, total_inj


def main():
    parser = argparse.ArgumentParser(description="Analyze inst_exec.log and merge results with previous CSV.")
    parser.add_argument("--app", "-a", required=True, help="Application name")
    parser.add_argument("--test", "-t", required=True, help="Test identifier", type=str)
    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst_exec.log")
    effects_occ, results_occ, params_by_pair = parse_log(log_path)

    out_path, total_masked, total_sdc, total_due, total_others, total_inj = write_csv(
        args.app, args.test, effects_occ, results_occ
    )

    # Persist invalid parameter combinations to skip in future rounds
    # Criterion: no valid writer/reader resolved -> recs contain a single record with src == 'invalid'
    invalid_keys = set()
    for inj_key, recs in effects_occ.items():
        # Pair key without the occurrence index
        run_id, name, _ = inj_key
        if len(recs) == 1 and (recs[0].get("src") == "invalid"):
            combo = params_by_pair.get((run_id, name))
            if combo:
                invalid_keys.add(combo)

    if invalid_keys:
        store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "invalid_param_combos.txt")
        existing = set()
        if os.path.exists(store_path):
            with open(store_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.add(line)
        new_items = sorted(k for k in invalid_keys if k not in existing)
        if new_items:
            with open(store_path, "a", encoding="utf-8") as f:
                for k in new_items:
                    f.write(k + "\n")
            print(f"Appended {len(new_items)} invalid parameter combinations to {store_path}")

    print(f"CSV written: {out_path}")
    print("========== Overall Summary (previous + this log) ==========")
    print(f" Total injections : {total_inj}")
    print(f"   Masked         : {total_masked}")
    print(f"   SDC            : {total_sdc}")
    print(f"   DUE            : {total_due}")
    print(f"   Others         : {total_others}")


if __name__ == "__main__":
    main()
