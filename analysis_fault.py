#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import shutil
import sys
import math
import random
from collections import defaultdict, Counter
from datetime import datetime
from copy import deepcopy

# -----------------------------
# 基础解析与工具
# -----------------------------

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
    """逐条解析日志（同时支持“内联 Effects+WRITER/READER”与“分段模式”）"""
    # 1) 独立的 Effects 起始行（旧格式）
    re_effects_start = re.compile(
        r"^\[Run\s+(\d+)\]\s+Effects from\s+(?:.+/)?(tmp\.out\d+):\s*$"
    )
    # 2) 内联形式（新格式）
    re_effects_inline = re.compile(
        r"^\[Run\s+(\d+)\]\s+Effects from\s+(?:.+/)?(tmp\.out\d+):\s*(.*\S.*)$"
    )
    # 3) Writer/Reader 本体
    re_writer = re.compile(
        r"^\[(?P<src>[-A-Za-z0-9_]+)_FI_WRITER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    re_reader = re.compile(
        r"^\[(?P<src>[-A-Za-z0-9_]+)_FI_READER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    # 4) 结果与参数
    re_result = re.compile(r"^\[Run\s+(\d+)\]\s+(tmp\.out\d+):\s*(.*?)\s*$")
    re_params = re.compile(r"^\[INJ_PARAMS\]\s+\[Run\s+(\d+)\]\s+(tmp\.out\d+)\s+(.*)$")

    latest_effects_by_pair = {}   # {(run_id,name): [records...]} 去重后的最新汇总
    params_by_pair = {}           # {(run_id,name): "k=v;..."}
    cur_key = None
    cur_writers, cur_readers = [], []

    occ_counter = defaultdict(int)
    effects_occ, results_occ = {}, {}

    def _merge_unique(writers, readers):
        """合并 WRITER 与 READER，并按 (src,kernel,line,text) 去重"""
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
                {"kernel": "invalid_summary", "inst_line": -1, "inst_text": "", "src": "invalid"}
            ]
        return merged

    def _merge_records(existing, add):
        if not existing:
            return _merge_unique(add, [])
        if not add:
            return _merge_unique(existing, [])
        return _merge_unique(existing + add, [])

    def flush_current_effects():
        nonlocal cur_key, cur_writers, cur_readers
        if cur_key is not None:
            new_pack = _merge_unique(cur_writers, cur_readers)
            existed = latest_effects_by_pair.get(cur_key, [])
            latest_effects_by_pair[cur_key] = _merge_records(existed, new_pack)
            cur_key = None
            cur_writers, cur_readers = [], []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")

                # (A) 内联 Effects + Writer/Reader
                m = re_effects_inline.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    rest = m.group(3).strip()
                    new_key = (run_id, name)

                    if cur_key != new_key:
                        flush_current_effects()
                        cur_key = new_key
                        cur_writers, cur_readers = [], []

                    mw = re_writer.match(rest)
                    if mw:
                        cur_writers.append(
                            {
                                "kernel": mw.group(2),
                                "inst_line": int(mw.group(4)),
                                "inst_text": mw.group(5).strip(),
                                "src": mw.group("src"),
                            }
                        )
                        continue
                    mr = re_reader.match(rest)
                    if mr:
                        cur_readers.append(
                            {
                                "kernel": mr.group(2),
                                "inst_line": int(mr.group(4)),
                                "inst_text": mr.group(5).strip(),
                                "src": mr.group("src"),
                            }
                        )
                        continue
                    continue

                # (B) 旧格式的 Effects 起始
                m = re_effects_start.match(line)
                if m:
                    flush_current_effects()
                    run_id = int(m.group(1))
                    name = m.group(2)
                    cur_key = (run_id, name)
                    cur_writers, cur_readers = [], []
                    continue

                # (C) 在当前 key 下累积 WRITER/READER（旧格式）
                if cur_key is not None:
                    m = re_writer.match(line)
                    if m:
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
                        cur_readers.append(
                            {
                                "kernel": m.group(2),
                                "inst_line": int(m.group(4)),
                                "inst_text": m.group(5).strip(),
                                "src": m.group("src"),
                            }
                        )
                        continue

                # (D) INJ_PARAMS
                m = re_params.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    params_by_pair[(run_id, name)] = m.group(3).strip()
                    continue

                # (E) 结果行：绑定结果
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
                        current_pack = _merge_unique(cur_writers, cur_readers)
                        existed = latest_effects_by_pair.get(pair, [])
                        recs = _merge_records(existed, current_pack)
                        latest_effects_by_pair[pair] = deepcopy(recs)
                    else:
                        recs = latest_effects_by_pair.get(
                            pair,
                            [
                                {"kernel": "invalid_summary", "inst_line": -1, "inst_text": "", "src": "invalid"}
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
    对“invalid 且结果为 SDC”的新注入：
      - 复制 tmp.outN 到 error_classification/
      - 同时复制 inst_exec.log 并重命名为 inst_exec_{...}.log
      - 记录到 error_classification/error_list.txt
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    err_dir = os.path.join(base_dir, "error_classification")
    os.makedirs(err_dir, exist_ok=True)
    list_path = os.path.join(err_dir, "error_list.txt")

    appended = 0
    lines_to_append = []

    for inj_key, recs in effects_occ.items():
        run_id, name, _ = inj_key
        is_invalid = (len(recs) == 1 and (recs[0].get("src") == "invalid" or recs[0].get("kernel") == "invalid_summary"))
        if not is_invalid:
            continue
        res = results_occ.get(inj_key, "Others")
        if res != "SDC":
            continue

        base_dir2 = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(base_dir2, f"logs{run_id}", name)
        if not os.path.exists(src_path):
            alt = os.path.join(f"logs{run_id}", name)
            if os.path.exists(alt):
                src_path = alt
            else:
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

        dest_name = f"{app}_{test}_{components}_{bitflip}_r{run_id}_{name}"
        dest_path = _ensure_unique_path(err_dir, dest_name)
        try:
            shutil.copyfile(src_path, dest_path)
            appended += 1
        except Exception:
            dest_path = "COPY_FAILED"

        inst_log_src = os.path.join(base_dir, "inst_exec.log")
        inst_log_dest_name = f"inst_exec_{app}_{test}_{components}_{bitflip}_r{run_id}_{name}.log"
        inst_log_dest_path = os.path.join(err_dir, inst_log_dest_name)
        try:
            if os.path.exists(inst_log_src):
                shutil.copyfile(inst_log_src, inst_log_dest_path)
        except Exception as e:
            print(f"WARN: failed to copy inst_exec.log for invalid-SDC: {e}")
            inst_log_dest_path = "COPY_FAILED"

        combo_full = params_by_pair.get((run_id, name), "") or ""
        combo_reduced = reduce_combo(combo_full) if combo_full else ""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_line = (
            f"time={ts}\tapp={app}\ttest={test}\tcomponent={components}\tbitflip={bitflip}\t"
            f"run={run_id}\ttmp={name}\tresult=SDC\tsrc_path={src_path}\tsaved_path={dest_path}\t"
            f"inst_exec_saved={inst_log_dest_path}\t"
            f"params={combo_full}\treduced_params={combo_reduced}"
        )
        lines_to_append.append(info_line)

    if lines_to_append:
        with open(list_path, "a", encoding="utf-8") as f:
            for line in lines_to_append:
                f.write(line + "\n")
        print(f"Captured {appended} invalid-SDC outputs to {err_dir} and logged {len(lines_to_append)} entries.")
    return appended, len(lines_to_append)


def write_csv(app: str, test: str, components: str, bitflip: str, effects_occ, results_occ, params_by_pair):
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

    # 合并旧 CSV
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

    # 本次注入结果
    for inj_key, recs in effects_occ.items():
        res_cat = results_occ.get(inj_key, "Others")
        run_id, name, _ = inj_key
        combo = params_by_pair.get((run_id, name), "") or ""

        # 提取 reg_name
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

            if kernel == "invalid_summary" or src == "invalid":
                continue
            if reg_names_this:
                for rn in reg_names_this:
                    regname_counts[key][rn] += 1

    # 写回 CSV（原子替换）
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
            for src in sorted(all_srcs):
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


# -----------------------------
# 仅新数据的 invalid 比例
# -----------------------------

def _compute_perc_inv_from_new(effects_occ):
    """
    Perc_inv（仅新数据）：新数据中 invalid 的 tot_inj / 新数据中所有行的 tot_inj
    （按 rec 粒度：一次注入若产生多个 reader，会计多次）
    """
    new_all = 0
    new_inv = 0
    for recs in effects_occ.values():
        for rec in recs:
            new_all += 1
            if rec.get("src") == "invalid" or rec.get("kernel") == "invalid_summary":
                new_inv += 1
    return (new_inv / new_all) if new_all > 0 else 0.0


# -----------------------------
# 工具：result_info 维护
# -----------------------------

def _append_summary_csv(info_path: str, header_fields, row_dict: dict):
    """把一行摘要追加到 result_info（若文件不存在则写表头）"""
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    write_header = not os.path.exists(info_path) or os.path.getsize(info_path) == 0
    mode = "a" if os.path.exists(info_path) else "w"
    with open(info_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_fields)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


def _infer_next_cycle_simple(info_path: str):
    """读取 result_info，推断下一轮编号（最后一行 cycle + 1），无则从 1 开始"""
    try:
        if not os.path.exists(info_path) or os.path.getsize(info_path) == 0:
            return 1
        with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            last_cycle = 0
            for row in reader:
                try:
                    last_cycle = int(row.get("cycle", "0") or "0")
                except Exception:
                    pass
        return (last_cycle + 1) if last_cycle >= 0 else 1
    except Exception:
        return 1


def _read_last_vals(info_path: str, k: int = 10):
    """读取最近 k 行的 (A,B)，从旧到新返回列表"""
    hist = []
    if os.path.exists(info_path) and os.path.getsize(info_path) > 0:
        with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    a = float(row.get("A", "nan"))
                    b = float(row.get("B", "nan"))
                    hist.append((a, b))
                except Exception:
                    continue
    return hist[-k:]


def _append_suffix_before_ext(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


# -----------------------------
# Spearman 相关系数工具
# -----------------------------

def _rankdata_avg_ties(values):
    """返回平均秩（1..n），处理并列。"""
    n = len(values)
    pairs = sorted((val, idx) for idx, val in enumerate(values))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        # 平均秩（1-based）
        rank_start = i + 1
        rank_end = j
        avg_rank = (rank_start + rank_end) / 2.0
        for k in range(i, j):
            ranks[pairs[k][1]] = avg_rank
        i = j
    return ranks


def _pearson_corr(x, y):
    """计算皮尔逊相关，长度一致；若方差为 0 则返回 0"""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _spearman_corr(x, y):
    """Spearman 相关 = Pearson(秩)；并列用平均秩"""
    rx = _rankdata_avg_ties(x)
    ry = _rankdata_avg_ties(y)
    return _pearson_corr(rx, ry)


# -----------------------------
# 新增：A/B = 旧 vs 合并（前60%交集）Spearman
# -----------------------------

def _aggregate_new_round_counts(effects_occ, results_occ):
    """
    按“非 invalid 指令”聚合本轮新增的 tot_inj 与 SDC 计数。
    key = (kernel, inst_line, inst_text)
    """
    new_tot = defaultdict(int)
    new_sdc = defaultdict(int)
    for inj_key, recs in effects_occ.items():
        res_cat = results_occ.get(inj_key, "Others")
        is_sdc = (res_cat == "SDC")
        for rec in recs:
            kernel = rec.get("kernel") or "unknown"
            inst_line = rec.get("inst_line")
            inst_line = -1 if inst_line is None else int(inst_line)
            inst_text = rec.get("inst_text") or "unknown"
            src = rec.get("src", "unknown")
            if kernel == "invalid_summary" or src == "invalid":
                continue
            k = (kernel, inst_line, inst_text)
            new_tot[k] += 1
            if is_sdc:
                new_sdc[k] += 1
    return new_tot, new_sdc


def _load_combined_from_csv(out_csv_path: str):
    """
    从合并后的 CSV 读取“非 invalid 指令”的 tot_inj 与 SDC。
    返回 dict: key -> (tot_inj, sdc)
    """
    comb_tot = {}
    comb_sdc = {}
    if not os.path.exists(out_csv_path):
        return comb_tot, comb_sdc
    with open(out_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel = row.get("kernel", "")
            if kernel == "invalid_summary":
                continue
            inst_line = -1 if row.get("inst_line", "") == "" else int(row.get("inst_line"))
            inst_text = row.get("inst_text", "")
            k = (kernel, inst_line, inst_text)
            try:
                tot = int(row.get("tot_inj", 0))
                sdc = int(row.get("SDC", 0))
            except Exception:
                tot = 0
                sdc = 0
            comb_tot[k] = tot
            comb_sdc[k] = sdc
    return comb_tot, comb_sdc


def _compute_metrics(tot_map: dict, sdc_map: dict):
    """
    根据 tot_map/sdc_map 计算：
      - 指标1：tot_inj / sum_tot_inj
      - 指标2：sdc / tot_inj
      - 分数：指标1 * 指标2
    仅对 tot_inj > 0 的键返回。
    """
    metrics1 = {}
    metrics2 = {}
    scores = {}
    sum_tot = sum(v for v in tot_map.values() if v > 0)
    if sum_tot <= 0:
        return metrics1, metrics2, scores
    for k, ti in tot_map.items():
        if ti <= 0:
            continue
        si = sdc_map.get(k, 0)
        m1 = ti / sum_tot
        m2 = (si / ti) if ti > 0 else 0.0
        metrics1[k] = m1
        metrics2[k] = m2
        scores[k] = m1 * m2
    return metrics1, metrics2, scores


def _top_keys_by_score(scores: dict, ratio: float):
    """
    按分数降序取前 ratio(0~1] 比例的键集合；遇到并列按 (kernel, line, text) 稳定次序打破。
    """
    if not scores:
        return set()
    n = len(scores)
    top_n = max(1, math.ceil(ratio * n))
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2]))
    return set(k for k, _ in ordered[:top_n])


def compute_A_B_old_vs_combined(out_csv_path: str, effects_occ, results_occ, top_ratio: float = 0.6):
    """
    A/B 计算（满足你的新规则）：
      1) 读合并后的 CSV => “合并(旧+新)”的 tot_inj/sdc。
      2) 用本轮新增（effects_occ/results_occ）聚合出“新” tot_inj/sdc。
      3) 旧 = 合并 − 新（逐指令、下界 0）。
      4) 在“旧”与“合并”中分别计算指标1、指标2与 score=指标1*指标2；
         各自取前 60%（top_ratio）集合，求交集 S；
         若 |S| < 2 => A=B=0。
      5) 在 S 上计算：
         A = Spearman( 合并.指标1[k], 旧.指标1[k] )
         B = Spearman( 合并.指标2[k], 旧.指标2[k] )
    """
    # 合并后的累计
    comb_tot, comb_sdc = _load_combined_from_csv(out_csv_path)

    # 本轮新增
    new_tot, new_sdc = _aggregate_new_round_counts(effects_occ, results_occ)

    # 旧 = 合并 - 新
    old_tot = {}
    old_sdc = {}
    for k, ti in comb_tot.items():
        old_tot[k] = max(0, ti - new_tot.get(k, 0))
        old_sdc[k] = max(0, comb_sdc.get(k, 0) - new_sdc.get(k, 0))

    # 计算指标
    m1_comb, m2_comb, scores_comb = _compute_metrics(comb_tot, comb_sdc)
    m1_old,  m2_old,  scores_old  = _compute_metrics(old_tot,  old_sdc)

    # 前 60% 集合 & 交集
    top_comb = _top_keys_by_score(scores_comb, top_ratio)
    top_old  = _top_keys_by_score(scores_old,  top_ratio)
    inter = top_comb & top_old

    if len(inter) < 2:
        return 0.0, 0.0, inter

    # Spearman
    m1_x = [m1_comb[k] for k in inter]
    m1_y = [m1_old[k]  for k in inter]
    m2_x = [m2_comb[k] for k in inter]
    m2_y = [m2_old[k]  for k in inter]

    A = _spearman_corr(m1_x, m1_y)
    B = _spearman_corr(m2_x, m2_y)
    return A, B, inter


# -----------------------------
# 辅助：汇总非 invalid 指令 tot_inj，并按阈值做分段备份
# -----------------------------

def _sum_noninvalid_totinj_from_csv(out_csv_path: str) -> int:
    """从累积 CSV 中计算所有非 invalid 行的 tot_inj 之和。"""
    if not os.path.exists(out_csv_path):
        return 0
    s = 0
    with open(out_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("kernel", "") == "invalid_summary":
                continue
            try:
                s += int(row.get("tot_inj", 0))
            except Exception:
                pass
    return s

def _count_noninvalid_rows_and_only_totinj(out_csv_path: str):
    """
    统计合并CSV中(新+旧)非 invalid_summary 行的数量；
    若仅有一条有效指令，返回 (1, 该行的 tot_inj)，否则返回 (数量, 0)。
    """
    if not os.path.exists(out_csv_path):
        return 0, 0
    cnt = 0
    only_totinj = 0
    with open(out_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("kernel", "") == "invalid_summary":
                continue
            cnt += 1
            if cnt == 1:
                try:
                    only_totinj = int(row.get("tot_inj", 0))
                except Exception:
                    only_totinj = 0
    return cnt, only_totinj



def _maybe_save_threshold_snapshots(out_csv_path: str, thresholds=(384, 600, 1067, 2401)):
    """
    当累积的非 invalid tot_inj 之和达到各阈值时，首次各自另存 CSV（_384/_600/_1067/_2401）。
    若快照已存在则保持不变（不覆盖）。
    """
    total_noninv = _sum_noninvalid_totinj_from_csv(out_csv_path)
    saved = []
    for thr in thresholds:
        snap = _append_suffix_before_ext(out_csv_path, f"_{thr}")
        if total_noninv >= thr and not os.path.exists(snap):
            try:
                shutil.copyfile(out_csv_path, snap)
                print(f"[Snapshot] non-invalid tot_inj={total_noninv} >= {thr}. Saved: {snap}")
                saved.append(snap)
            except Exception as e:
                print(f"[WARN] Failed to save snapshot {snap}: {e}")
    return total_noninv, saved


# -----------------------------
# 主流程（新停机条件 & 新 A/B）
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze inst_exec.log and merge results with previous CSV; compute Spearman A/B (old vs combined, top-60%) & manage snapshots."
    )
    parser.add_argument("--app", "-a", required=True, help="Application name")
    parser.add_argument("--test", "-t", required=True, help="Test identifier", type=str)
    parser.add_argument("--component", "-c", required=True, help="Component set")
    parser.add_argument("--bitflip", "-b", required=True, help="Number of bit flips to inject")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "inst_exec.log")

    # 解析本次日志
    effects_occ, results_occ, params_by_pair = parse_log(log_path)

    # 计算 Perc_inv（仅新数据）
    perc_inv_new = _compute_perc_inv_from_new(effects_occ)

    # 写入/合并结果 CSV
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_result_{args.app}_{args.test}_{args.component}_{args.bitflip}.csv")

    out_path, total_masked, total_sdc, total_due, total_others, total_inj = write_csv(
        args.app, args.test, args.component, args.bitflip, effects_occ, results_occ, params_by_pair
    )

    # 记录 invalid 参数组合（保持原功能）
    invalid_keys = set()
    for inj_key, recs in effects_occ.items():
        run_id, name, _ = inj_key
        if len(recs) == 1 and (recs[0].get("src") == "invalid"):
            combo = params_by_pair.get((run_id, name))
            if combo:
                invalid_keys.add(reduce_combo(combo))

    if invalid_keys:
        store_path = os.path.join(base_dir, "invalid_param_combos.txt")
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

    # 对“invalid 且 SDC”的新注入，拷贝 tmp.out 并记录信息（保持原功能）
    _collect_invalid_sdc_outputs(
        effects_occ,
        results_occ,
        params_by_pair,
        app=args.app,
        test=args.test,
        components=args.component,
        bitflip=args.bitflip,
    )

    # ===== 新：A/B（旧 vs 合并，前60%交集）=====
    A, B, _inter = compute_A_B_old_vs_combined(out_path, effects_occ, results_occ, top_ratio=0.6)

    # 写入 result_info（cycle, A, B, Perc_inv）
    info_dir = "result_info"
    os.makedirs(info_dir, exist_ok=True)
    info_path = os.path.join(info_dir, f"result_info_{args.app}_{args.test}_{args.component}_{args.bitflip}.csv")
    header_fields = ["cycle", "A", "B", "Perc_inv"]

    cycle = _infer_next_cycle_simple(info_path)
    _append_summary_csv(
        info_path,
        header_fields,
        {
            "cycle": cycle,
            "A": f"{A:.6f}",
            "B": f"{B:.6f}",
            "Perc_inv": f"{perc_inv_new:.6f}",
        },
    )

    # 控制台打印
    print("========== Round Summary ==========")
    print(f" Cycle={cycle} | A={A:.6f} | B={B:.6f} | Perc_inv (new only)={perc_inv_new:.6f}")
    print(f" Totals  Masked: {total_masked} | SDC: {total_sdc} | DUE: {total_due} | Others: {total_others} | All: {total_inj}")
    # ---- 新停机规则：到第三轮时，(新+旧) 仅存在 1 条有效指令，且其 tot_inj > 384 则停机 ----
    if cycle == 3:
        valid_cnt, only_totinj = _count_noninvalid_rows_and_only_totinj(out_path)
        if valid_cnt == 1 and only_totinj > 384:
            print(f"[EXIT] Round-3 rule hit: only 1 valid instruction with tot_inj={only_totinj} (>384). exit99.")
            sys.exit(99)


    # 读取最近历史，计算“连续 >=0.98”的计数（新的停机阈值）
    hist = _read_last_vals(info_path, k=10)  # 取最近最多 10 轮
    def streak_ge(vals, thr=0.98):
        cnt = 0
        for v in reversed(vals):
            if v >= thr:
                cnt += 1
            else:
                break
        return cnt

    As = [ab[0] for ab in hist]
    Bs = [ab[1] for ab in hist]
    A_streak = streak_ge(As, 0.98)
    B_streak = streak_ge(Bs, 0.98)

    # ---- 停机判断优先（若停机则不再执行分段里程碑快照）----
    if A_streak >= 3 and B_streak >= 3:
        print(f"[EXIT] A and B have both been >=0.98 for 3 consecutive rounds (A_streak={A_streak}, B_streak={B_streak}). exit99.")
        sys.exit(99)

    # A 连续>=3 且 B 尚未连续>=3：另存 test_result_*_A.csv  （A/B 判断已使用新 A/B，阈值=0.98）
    snap_A_path = _append_suffix_before_ext(out_path, "_A")
    if A_streak >= 3 and B_streak < 3:
        try:
            shutil.copyfile(out_path, snap_A_path)
            print(f"[Info] A has >=0.98 for {A_streak} consecutive rounds (B_streak={B_streak}). Snapshot saved: {snap_A_path}")
        except Exception as e:
            print(f"[WARN] Failed to save snapshot {snap_A_path}: {e}")

    # 若 A 在曾经达到连续>=3 后又跌破（即当前连续计数 < 3），删除 _A 快照（若存在）
    if A_streak < 3 and os.path.exists(snap_A_path):
        try:
            os.remove(snap_A_path)
            print(f"[Info] A-streak dropped below 3 (>=0.98). Snapshot removed: {snap_A_path}")
        except Exception as e:
            print(f"[WARN] Failed to remove {snap_A_path}: {e}")

    # ---- 分段里程碑快照（保持原逻辑，仅在“未停机”的情况下评估并保存；且仅首次保存）----
    total_noninv, saved_snaps = _maybe_save_threshold_snapshots(out_path, thresholds=(384, 600, 1067, 2401))
    if saved_snaps:
        print(f"[Info] Non-invalid tot_inj sum = {total_noninv}. Saved milestone snapshots: {', '.join(saved_snaps)}")

    # 正常结束
    return


if __name__ == "__main__":
    main()
