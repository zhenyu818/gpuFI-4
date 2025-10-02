#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import shutil
import sys
import math
from collections import defaultdict, Counter
from datetime import datetime
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
    re_effects_start = re.compile(
        r"^\[Run\s+(\d+)\]\s+Effects from\s+(?:.+/)?(tmp\.out\d+):\s*$"
    )
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

    def _merge_unique(writers, readers):
        # 合并 WRITER 与 READER，并按 (src,kernel,line,text) 去重
        seen = set()
        merged = []
        for rec in writers + readers:
            key = (rec.get("src"), rec.get("kernel"), rec.get("inst_line"), rec.get("inst_text"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(deepcopy(rec))
        if not merged:
            merged = [
                {
                    "kernel": "invalid_summary",
                    "inst_line": -1,
                    "inst_text": "",
                    "src": "invalid",
                }
            ]
        return merged

    def flush_current_effects():
        nonlocal cur_key, cur_writers, cur_readers
        if cur_key is not None:
            latest_effects_by_pair[cur_key] = _merge_unique(cur_writers, cur_readers)
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
                        # 支持一次注入打印多个 FI_WRITER：累积
                        cur_writers.append(
                            {
                                "kernel": m.group(2),
                                "inst_line": int(m.group(4)),
                                "inst_text": m.group(5).strip(),
                                "src": m.group("src"),
                            }
                        )
                        continue
                    m = re_reader.match(line)
                    if m:
                        # 累积所有 FI_READER（与 FI_WRITER 并存）
                        cur_readers.append(
                            {
                                "kernel": m.group(2),
                                "inst_line": int(m.group(4)),
                                "inst_text": m.group(5).strip(),
                                "src": m.group("src"),
                            }
                        )
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
                        recs = _merge_unique(cur_writers, cur_readers)
                        latest_effects_by_pair[pair] = deepcopy(recs)
                    else:
                        recs = latest_effects_by_pair.get(
                            pair,
                            [
                                {
                                    "kernel": "invalid_summary",
                                    "inst_line": -1,
                                    "inst_text": "",
                                    "src": "invalid",
                                }
                            ],
                        )

                    effects_occ[inj_key] = deepcopy(recs)
                    results_occ[inj_key] = res
                    continue

        flush_current_effects()

    except FileNotFoundError:
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    return effects_occ, results_occ, params_by_pair


def reduce_combo(combo: str) -> str:
    """规约 key=val;key=val;... 至稳定字段序，便于去重存档"""
    keep_order = [
        "comp",
        "per_warp",
        "kernel",
        "thread",
        "warp",
        "block",
        "cycle",
        "reg_name",
        "reg_rand_n",
    ]
    kv = {}
    for part in combo.split(";"):
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        kv[k] = v
    return ";".join([f"{k}={kv[k]}" for k in keep_order if k in kv])


def _ensure_unique_path(dest_dir: str, base_name: str) -> str:
    """Return a non-clobbering destination path by appending _N if needed."""
    candidate = os.path.join(dest_dir, base_name)
    if not os.path.exists(candidate):
        return candidate
    root, ext = os.path.splitext(base_name)
    n = 1
    while True:
        cand = os.path.join(dest_dir, f"{root}_{n}{ext}")
        if not os.path.exists(cand):
            return cand
        n += 1


def _collect_invalid_sdc_outputs(effects_occ, results_occ, params_by_pair, app, test, components, bitflip):
    """
    For each newly parsed injection: if it's classified as invalid but the result is SDC,
    copy the corresponding tmp.out file from logs<run_id>/tmp.outN to error_classification,
    and append a record into error_classification/error_list.txt.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    err_dir = os.path.join(base_dir, "error_classification")
    os.makedirs(err_dir, exist_ok=True)
    list_path = os.path.join(err_dir, "error_list.txt")

    appended = 0
    lines_to_append = []

    for inj_key, recs in effects_occ.items():
        run_id, name, _ = inj_key
        # invalid if sentinel only
        is_invalid = (len(recs) == 1 and (recs[0].get("src") == "invalid" or recs[0].get("kernel") == "invalid_summary"))
        if not is_invalid:
            continue
        res = results_occ.get(inj_key, "Others")
        if res != "SDC":
            continue

        # source tmp.out path (logs<run_id>/tmp.outN)
        src_path = os.path.join(base_dir, f"logs{run_id}", name)
        if not os.path.exists(src_path):
            # try a conservative fallback without base_dir (shouldn't be needed)
            alt = os.path.join(f"logs{run_id}", name)
            if os.path.exists(alt):
                src_path = alt
            else:
                # cannot find the tmp.out; skip but still note it
                combo_full = params_by_pair.get((run_id, name), "") or ""
                combo_reduced = reduce_combo(combo_full) if combo_full else ""
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                info_line = (
                    f"time={ts}\tapp={app}\ttest={test}\tcomponent={components}\tbitflip={bitflip}\t"
                    f"run={run_id}\ttmp={name}\tresult=SDC\tsrc_path=NOT_FOUND\tsaved_path=SKIPPED\t"
                    f"params={combo_full}\treduced_params={combo_reduced}"
                )
                lines_to_append.append(info_line)
                continue

        # destination name: include context to avoid collisions
        dest_name = f"{app}_{test}_{components}_{bitflip}_r{run_id}_{name}"
        dest_path = _ensure_unique_path(err_dir, dest_name)
        try:
            shutil.copyfile(src_path, dest_path)
            appended += 1
        except Exception:
            # fall back to noting copy failure but still append info
            dest_path = "COPY_FAILED"

        combo_full = params_by_pair.get((run_id, name), "") or ""
        combo_reduced = reduce_combo(combo_full) if combo_full else ""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_line = (
            f"time={ts}\tapp={app}\ttest={test}\tcomponent={components}\tbitflip={bitflip}\t"
            f"run={run_id}\ttmp={name}\tresult=SDC\tsrc_path={src_path}\tsaved_path={dest_path}\t"
            f"params={combo_full}\treduced_params={combo_reduced}"
        )
        lines_to_append.append(info_line)

    if lines_to_append:
        with open(list_path, "a", encoding="utf-8") as f:
            for line in lines_to_append:
                f.write(line + "\n")
        print(f"Captured {appended} invalid-SDC outputs to {err_dir} and logged {len(lines_to_append)} entries.")
    return appended, len(lines_to_append)


def write_csv(app: str, test: str,components: str, bitflip: str, effects_occ, results_occ, params_by_pair):
    """
    生成/合并 CSV，并为每条指令新增 reg_names 列。
    规则：
      - 对于 kernel == "invalid_summary" 的行（即 invalid 行），不打印 reg_names，且不累计其 reg_name 统计。
      - 合并旧 CSV 时也忽略 invalid 行的 reg_names。
    """
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_result_{app}_{test}_{components}_{bitflip}.csv")

    inst_counts = defaultdict(
        lambda: defaultdict(lambda: {"Masked": 0, "SDC": 0, "DUE": 0, "Others": 0})
    )
    all_srcs = set()
    regname_counts = defaultdict(Counter)  # 仅非 invalid 行统计

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
                    if col in ("kernel", "inst_line", "inst_text", "reg_names"):
                        continue
                    if col.endswith(("_Masked", "_SDC", "_DUE", "_Others")):
                        src, cat = col.rsplit("_", 1)
                        inst_counts[key][src][cat] += int(val)
                        all_srcs.add(src)

                if kernel != "invalid_summary":
                    rn_field = row.get("reg_names", "").strip()
                    if rn_field:
                        for token in rn_field.split(","):
                            token = token.strip()
                            if not token or ":" not in token:
                                continue
                            rn, cnt = token.split(":", 1)
                            rn = rn.strip()
                            try:
                                regname_counts[key][rn] += int(cnt)
                            except ValueError:
                                pass

    # === 本次注入结果 ===
    for inj_key, recs in effects_occ.items():
        res_cat = results_occ.get(inj_key, "Others")
        run_id, name, _ = inj_key
        combo = params_by_pair.get((run_id, name), "") or ""

        # 提取 reg_name（可能是 "%r1" 或 "%r1:%p5:%rd7"）
        reg_names_this = []
        for part in combo.split(";"):
            part = part.strip()
            if part.startswith("reg_name="):
                raw = part.split("=", 1)[1].strip()
                if raw:
                    reg_names_this = [x.strip() for x in raw.split(":") if x.strip()]
                break

        for rec in recs:
            kernel = rec.get("kernel") or "unknown"
            inst_line = rec.get("inst_line") if rec.get("inst_line") is not None else -1
            inst_text = rec.get("inst_text") or "unknown"
            src = rec.get("src", "unknown")
            key = (kernel, inst_line, inst_text)

            inst_counts[key][src][res_cat] += 1
            all_srcs.add(src)

            # invalid 行或 src==invalid 不累计 reg_name
            if kernel == "invalid_summary" or src == "invalid":
                continue
            if reg_names_this:
                for rn in reg_names_this:
                    regname_counts[key][rn] += 1

    # === 写回 CSV（原子替换） ===
    src_columns = []
    for src in sorted(all_srcs):
        src_columns += [f"{src}_Masked", f"{src}_SDC", f"{src}_DUE", f"{src}_Others"]

    fieldnames = (
        ["kernel", "inst_line", "inst_text", "reg_names"]
        + src_columns
        + ["Masked", "SDC", "DUE", "Others", "tot_inj"]
    )

    total_masked = total_sdc = total_due = total_others = 0
    out_path_tmp = out_path + ".tmp"

    with open(out_path_tmp, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for kernel, inst_line, inst_text in sorted(
            inst_counts.keys(), key=lambda k: (k[0], k[1], k[2])
        ):
            src_map = inst_counts[(kernel, inst_line, inst_text)]
            row = {
                "kernel": kernel,
                "inst_line": "" if inst_line < 0 else inst_line,
                "inst_text": inst_text,
            }

            if kernel == "invalid_summary":
                row["reg_names"] = ""
            else:
                rn_counts = regname_counts.get((kernel, inst_line, inst_text), {})
                if rn_counts:
                    pairs = sorted(rn_counts.items(), key=lambda x: (-x[1], x[0]))
                    row["reg_names"] = ",".join([f"{rn}:{cnt}" for rn, cnt in pairs])
                else:
                    row["reg_names"] = ""

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

    os.replace(out_path_tmp, out_path)

    total_inj = total_masked + total_sdc + total_due + total_others
    return out_path, total_masked, total_sdc, total_due, total_others, total_inj


# ============= Perc_inv（仅新数据） =============

def _compute_perc_inv_from_new(effects_occ):
    """
    基于“本次解析得到的新数据”（effects_occ），计算 Perc_inv：
        Perc_inv = 新数据中 invalid 的 tot_inj / 新数据中所有行的 tot_inj
    注意：与 write_csv 一致，按“recs 粒度”计数（一次注入若产生多个 reader，会计多次）。
    """
    new_all = 0
    new_inv = 0
    for recs in effects_occ.values():
        for rec in recs:
            new_all += 1
            if rec.get("src") == "invalid" or rec.get("kernel") == "invalid_summary":
                new_inv += 1
    return (new_inv / new_all) if new_all > 0 else 0.0


# ============= CSV 汇总读取（供新指标与阈值判定使用） =============

def _read_csv_summary(path):
    """
    读取 CSV，返回：
      - rows_raw: 列表[dict]，按行原样（含 kernel/inst_text/tot_inj 等）
      - all_tot: 全部行 tot_inj 之和
    """
    rows_raw = []
    all_tot = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_raw.append(row)
            try:
                all_tot += int(row.get("tot_inj", "0"))
            except ValueError:
                pass
    return rows_raw, all_tot


def _compute_ratio_tot384_excluding_unknown(rows_raw):
    """
    计算：在当前 CSV（合并后）中，排除 invalid_summary 与 unknown，
    满足 tot_inj >= 384 的指令数 / 有记录指令总数 的比例。
    """
    valid_rows = []
    for r in rows_raw:
        kernel = r.get("kernel", "")
        inst_text = r.get("inst_text", "")
        if kernel == "invalid_summary":
            continue
        if (kernel or "").strip().lower() == "unknown":
            continue
        if (inst_text or "").strip().lower() == "unknown":
            continue
        valid_rows.append(r)

    denom = len(valid_rows)
    if denom == 0:
        return 0.0, 0, 0

    num = 0
    for r in valid_rows:
        try:
            tot = int(r.get("tot_inj", "0"))
        except ValueError:
            tot = 0
        if tot >= 384:
            num += 1

    return num / denom, num, denom


# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze inst_exec.log and merge results with previous CSV."
    )
    parser.add_argument("--app", "-a", required=True, help="Application name")
    parser.add_argument("--test", "-t", required=True, help="Test identifier", type=str)
    parser.add_argument("--component", "-c", required=True, help="Component set")
    parser.add_argument("--bitflip", "-b", required=True, help="Number of bit flips to inject")
    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst_exec.log")

    # 解析本次日志
    effects_occ, results_occ, params_by_pair = parse_log(log_path)

    # Perc_inv（仅新数据）
    perc_inv_new = _compute_perc_inv_from_new(effects_occ)

    # 结果 CSV 路径
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_result_{args.app}_{args.test}_{args.component}_{args.bitflip}.csv")

    # 写入合并后的结果
    out_path, total_masked, total_sdc, total_due, total_others, total_inj = write_csv(
        args.app, args.test, args.component, args.bitflip, effects_occ, results_occ, params_by_pair)

    # —— 记录 invalid 参数组合（保持原有功能）——
    invalid_keys = set()
    for inj_key, recs in effects_occ.items():
        run_id, name, _ = inj_key
        if len(recs) == 1 and (recs[0].get("src") == "invalid"):
            combo = params_by_pair.get((run_id, name))
            if combo:
                invalid_keys.add(reduce_combo(combo))

    if invalid_keys:
        store_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "invalid_param_combos.txt"
        )
        existing = set()
        if os.path.exists(store_path):
            with open(store_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.add(line)
        new_items = sorted(k for k in invalid_keys if k and k not in existing)
        if new_items:
            with open(store_path, "a", encoding="utf-8") as f:
                for k in new_items:
                    f.write(k + "\n")
            print(
                f"Appended {len(new_items)} invalid parameter combinations to {store_path}"
            )

    # 对“被归为 invalid 但结果为 SDC”的新注入，拷贝 tmp.out 并记录信息（保持原有功能）
    _collect_invalid_sdc_outputs(
        effects_occ,
        results_occ,
        params_by_pair,
        app=args.app,
        test=args.test,
        components=args.component,
        bitflip=args.bitflip,
    )

    # ===== 新指标与新的停止/备份条件 =====
    rows_raw, all_tot_sum = _read_csv_summary(out_path)

    # 比例：tot_inj >= 384 的指令数 / 有记录的指令数（排除 unknown 与 invalid_summary）
    ratio_384, cnt_ge384, cnt_total_valid = _compute_ratio_tot384_excluding_unknown(rows_raw)

    # 若所有指令的 tot_inj 之和 >= 384，则备份一个 *_384.csv（若不存在时）
    if all_tot_sum >= 384:
        backup_384 = out_path.replace(".csv", "_384.csv")
        if not os.path.exists(backup_384):
            shutil.copyfile(out_path, backup_384)
            print(f"Total tot_inj across all recorded instructions >= 384. Backup created: {backup_384}")

    # 若 ≥70% 的指令 tot_inj >= 384，则退出 99
    if cnt_total_valid > 0 and ratio_384 >= 0.70:
        print("========== Threshold Reached ==========")
        print(f" Instructions with tot_inj >= 384 : {cnt_ge384}/{cnt_total_valid} ({ratio_384:.6f})")
        print(" Condition met: 70% of recorded (excluding unknown) instructions have tot_inj >= 384.")
        sys.exit(99)

    # 保持原有控制台输出 + 新指标
    print(f"CSV written: {out_path}")
    print("========== Overall Summary (previous + this log) ==========")
    print(f" Total injections : {total_inj}")
    print(f"   Masked         : {total_masked}")
    print(f"   SDC            : {total_sdc}")
    print(f"   DUE            : {total_due}")
    print(f"   Others         : {total_others}")
    print("========== New Metrics ==========")
    print(f" Perc_inv (new only) : {perc_inv_new:.6f}")
    print("========== CSV Metrics ==========")
    print(f" tot_inj>=384 ratio (excluding unknown) : {cnt_ge384}/{cnt_total_valid} ({ratio_384:.6f})")
    print(f" Sum of tot_inj across all instructions : {all_tot_sum}")
    # 新增写入 result_info
    info_dir = "result_info"
    os.makedirs(info_dir, exist_ok=True)
    info_path = os.path.join(info_dir, f"result_info_{args.app}_{args.test}_{args.component}_{args.bitflip}.csv")

    file_exists = os.path.exists(info_path)
    with open(info_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Perc_inv_new", "cnt_ge384", "cnt_total_valid", "ratio_384", "all_tot_sum"])
        writer.writerow([
            f"{perc_inv_new:.6f}",
            cnt_ge384,
            cnt_total_valid,
            f"{ratio_384:.6f}",
            all_tot_sum
        ])
    print(f"Result info appended: {info_path}")


if __name__ == "__main__":
    main()
