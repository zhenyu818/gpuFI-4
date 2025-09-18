#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import sys
import math
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


def reduce_combo(combo: str) -> str:
    """规约 key=val;key=val;... 至稳定字段序，便于去重存档"""
    keep_order = [
        "comp", "per_warp", "kernel",
        "thread", "warp", "block", "cycle",
        "reg_name", "reg_rand_n",
    ]
    kv = {}
    for part in combo.split(";"):
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip(); v = v.strip()
        kv[k] = v
    return ";".join([f"{k}={kv[k]}" for k in keep_order if k in kv])


def write_csv(app: str, test: str, effects_occ, results_occ, params_by_pair):
    """
    生成/合并 CSV，并为每条指令新增 reg_names 列。
    规则：
      - 对于 kernel == "invalid_summary" 的行（即 invalid 行），不打印 reg_names，且不累计其 reg_name 统计。
      - 合并旧 CSV 时也忽略 invalid 行的 reg_names。
    """
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_result_{app}_{test}.csv")

    inst_counts = defaultdict(lambda: defaultdict(lambda: {"Masked": 0, "SDC": 0, "DUE": 0, "Others": 0}))
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

    fieldnames = ["kernel", "inst_line", "inst_text", "reg_names"] + src_columns + [
        "Masked", "SDC", "DUE", "Others", "tot_inj"
    ]

    total_masked = total_sdc = total_due = total_others = 0
    out_path_tmp = out_path + ".tmp"

    with open(out_path_tmp, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (kernel, inst_line, inst_text) in sorted(inst_counts.keys(), key=lambda k: (k[0], k[1], k[2])):
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


# ============= 新增：计算“本次新数据”的 Perc_inv（不看合并后） =============

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


# ================= 新增：Spearman 及结果记录相关工具 =================

def _rankdata_avg(values):
    """平均秩（ties 取平均秩），1-based ranks"""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _pearsonr(x, y):
    """皮尔逊相关，返回 None 表示不可定义"""
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = 0.0
    sx = 0.0
    sy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        num += dx * dy
        sx += dx * dx
        sy += dy * dy
    if sx <= 0.0 or sy <= 0.0:
        return None
    return num / math.sqrt(sx * sy)


def _spearmanr(x, y):
    """斯皮尔曼相关（皮尔逊相关作用在秩上）"""
    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    return _pearsonr(rx, ry)


def _read_csv_summary(path):
    """
    读取 CSV，返回：
      - noninv: dict[(kernel, inst_line, inst_text)] -> (tot_inj:int, SDC:int)，仅非 invalid
      - inv_tot: invalid 行 tot_inj 之和
      - all_tot: 全部行 tot_inj 之和
    """
    noninv = {}
    inv_tot = 0
    all_tot = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel = row["kernel"]
            inst_line = -1 if row["inst_line"] == "" else int(row["inst_line"])
            inst_text = row["inst_text"]
            try:
                tot_inj = int(row.get("tot_inj", "0"))
            except ValueError:
                tot_inj = 0
            try:
                sdc = int(row.get("SDC", "0"))
            except ValueError:
                sdc = 0
            all_tot += tot_inj
            if kernel == "invalid_summary":
                inv_tot += tot_inj
            else:
                noninv[(kernel, inst_line, inst_text)] = (tot_inj, sdc)
    return noninv, inv_tot, all_tot


def _format_val(v):
    if v is None:
        return "NA"
    try:
        return f"{float(v):.6f}"
    except Exception:
        return "NA"


def _append_result_info(app, test, sp_tot, sp_sdc, perc_inv):
    """
    追加写入 result_info/result_info_{app}_{test}.csv
    列：index,Sp_tot,Sp_SDC,Perc_inv
    返回：写入后的所有行（含表头）
    """
    info_dir = "result_info"
    os.makedirs(info_dir, exist_ok=True)
    info_path = os.path.join(info_dir, f"result_info_{app}_{test}.csv")

    rows = []
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))

    if not rows:
        rows = [["index", "Sp_tot", "Sp_SDC", "Perc_inv"]]

    try:
        last_idx = int(rows[-1][0]) if rows[-1][0].isdigit() else len(rows) - 1
    except Exception:
        last_idx = len(rows) - 1
    next_idx = last_idx + 1

    new_row = [str(next_idx), _format_val(sp_tot), _format_val(sp_sdc), _format_val(perc_inv)]
    rows.append(new_row)

    with open(info_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    print(f"Result info appended: {info_path}  ->  {','.join(new_row)}")
    return rows


def _check_consecutive_3(rows, thr=0.9):
    """
    只检查“最近3次”（最后三行数据，不含表头）是否同时满足：
      Sp_tot >= thr 且 Sp_SDC >= thr
    若不足3条数据或存在 NA/解析失败，返回 False
    """
    # rows 可能含表头：["index","Sp_tot","Sp_SDC","Perc_inv"]
    data_rows = rows[1:] if rows and rows[0] and rows[0][0] == "index" else rows
    if len(data_rows) < 3:
        return False

    last3 = data_rows[-3:]

    def geq(x):
        try:
            return float(x) >= thr
        except Exception:
            return False  # 包含 "NA" 等情况则判为不满足

    # 逐行同时满足 Sp_tot 和 Sp_SDC
    return all(geq(r[1]) and geq(r[2]) for r in last3)



# ================= 主流程 =================

def main():
    parser = argparse.ArgumentParser(description="Analyze inst_exec.log and merge results with previous CSV.")
    parser.add_argument("--app", "-a", required=True, help="Application name")
    parser.add_argument("--test", "-t", required=True, help="Test identifier", type=str)
    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst_exec.log")

    # 解析本次日志
    effects_occ, results_occ, params_by_pair = parse_log(log_path)

    # —— 新增：先基于“新数据”计算 Perc_inv（不看旧数据、也不看合并结果）——
    perc_inv_new = _compute_perc_inv_from_new(effects_occ)

    # 结果 CSV 路径
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_result_{args.app}_{args.test}.csv")

    # 读取旧结果（B），若存在
    had_old = os.path.exists(out_path)
    B_noninv = {}
    if had_old:
        try:
            B_noninv, _, _ = _read_csv_summary(out_path)
        except FileNotFoundError:
            had_old = False

    # 写入合并后的结果（A）
    out_path, total_masked, total_sdc, total_due, total_others, total_inj = write_csv(
        args.app, args.test, effects_occ, results_occ, params_by_pair
    )

    # 读取合并后的 A（仅用于 Sp_* 和其他打印；Perc_inv 用新数据的 perc_inv_new）
    A_noninv, A_inv_tot, A_all_tot = _read_csv_summary(out_path)

    # 若有旧结果，计算 Spearman(A vs B)，仅非 invalid 指令
    sp_tot = None
    sp_sdc = None
    if had_old:
        keys = set(A_noninv.keys()) | set(B_noninv.keys())
        A_tot_list, B_tot_list = [], []
        A_sdc_list, B_sdc_list = [], []
        for k in sorted(keys):
            a_tot, a_sdc = A_noninv.get(k, (0, 0))
            b_tot, b_sdc = B_noninv.get(k, (0, 0))
            A_tot_list.append(float(a_tot))
            B_tot_list.append(float(b_tot))
            A_sdc_list.append(float(a_sdc))
            B_sdc_list.append(float(b_sdc))
        sp_tot = _spearmanr(A_tot_list, B_tot_list)
        sp_sdc = _spearmanr(A_sdc_list, B_sdc_list)

    # —— 原有功能：记录 invalid 参数组合（使用 reduce_combo）——
    invalid_keys = set()
    for inj_key, recs in effects_occ.items():
        run_id, name, _ = inj_key
        if len(recs) == 1 and (recs[0].get("src") == "invalid"):
            combo = params_by_pair.get((run_id, name))
            if combo:
                invalid_keys.add(reduce_combo(combo))

    if invalid_keys:
        store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "invalid_param_combos.txt")
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
            print(f"Appended {len(new_items)} invalid parameter combinations to {store_path}")

    # 记录三项指标：Sp_tot、Sp_SDC、Perc_inv(本次新数据)
    rows_all = _append_result_info(args.app, args.test, sp_tot, sp_sdc, perc_inv_new)

    # 检查是否已连续 3 次 > 0.9（如满足则退出 99）
    if _check_consecutive_3(rows_all):
        print("满足条件：Sp_tot 与 Sp_SDC 已连续 3 次 > 0.9，停止新增注入。")
        sys.exit(99)

    # 保持原有控制台输出
    print(f"CSV written: {out_path}")
    print("========== Overall Summary (previous + this log) ==========")
    print(f" Total injections : {total_inj}")
    print(f"   Masked         : {total_masked}")
    print(f"   SDC            : {total_sdc}")
    print(f"   DUE            : {total_due}")
    print(f"   Others         : {total_others}")
    # 同时也把三项新指标打印一下
    print("========== New Metrics ==========")
    print(f" Sp_tot   : {_format_val(sp_tot)}")
    print(f" Sp_SDC   : {_format_val(sp_sdc)}")
    print(f" Perc_inv (new only) : {_format_val(perc_inv_new)}")


if __name__ == "__main__":
    main()
