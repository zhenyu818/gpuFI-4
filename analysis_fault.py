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

        src_path = os.path.join(base_dir, f"logs{run_id}", name)
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
# 新增：分半计算 + Spearman 相关系数
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


def compute_A_B_with_random_split(effects_occ, results_occ):
    """
    仅基于“本轮新数据（effects_occ/results_occ）”：
      - 将所有“非 invalid 的 rec”随机打乱后均分为两半；
      - 在每一半内，按“指令( kernel, inst_line, inst_text )”聚合 tot_inj 与 SDC；
      - 指标1 = tot_inj / (该半内所有非 invalid 指令的 tot_inj 之和)；
        指标2 = SDC / tot_inj( tot_inj=0 -> 0 )；
      - 以 score = 指标1*指标2 排序，分别取各自 Top 50% 的指令集合；
      - 在“两个 Top-50% 集合的交集”上，分别对指标1和指标2计算 Spearman 相关，得到 A、B。
    备注：若交集大小 < 2，则相关系数置 0.0。
    """
    # 1) 展平“本轮新数据”的 rec 列表（过滤 invalid）
    records = []  # 每个元素： (key, is_sdc)
    for inj_key, recs in effects_occ.items():
        res_cat = results_occ.get(inj_key, "Others")
        is_sdc = (res_cat == "SDC")
        for rec in recs:
            kernel = rec.get("kernel") or "unknown"
            inst_line = rec.get("inst_line")
            inst_line = -1 if inst_line is None else int(inst_line)
            inst_text = rec.get("inst_text") or "unknown"
            src = rec.get("src", "unknown")
            # 过滤 invalid
            if kernel == "invalid_summary" or src == "invalid":
                continue
            key = (kernel, inst_line, inst_text)
            records.append((key, is_sdc))

    # 若本轮没有非 invalid 数据，返回 0
    if not records:
        return 0.0, 0.0, set()

    # 2) 随机打乱并均分为两半
    random.shuffle(records)
    mid = len(records) // 2
    half1 = records[:mid]
    half2 = records[mid:]

    # 3) 两半内按指令聚合 tot_inj / SDC
    def agg_half(recs):
        tot = defaultdict(int)
        sdc = defaultdict(int)
        for k, is_sdc in recs:
            tot[k] += 1
            if is_sdc:
                sdc[k] += 1
        return tot, sdc

    tot1, sdc1 = agg_half(half1)
    tot2, sdc2 = agg_half(half2)

    # 指令全集（出现在任意半中的指令）
    keys = set(list(tot1.keys()) + list(tot2.keys()))
    if not keys:
        return 0.0, 0.0, set()

    # 4) 计算每半的指标与分数
    sum_tot1 = sum(tot1.values())
    sum_tot2 = sum(tot2.values())

    def calc_metrics_for_half(tot_map, sdc_map, sum_tot):
        metrics = {}
        for k in keys:
            ti = tot_map.get(k, 0)
            si = sdc_map.get(k, 0)
            m1 = (ti / sum_tot) if sum_tot > 0 else 0.0
            m2 = (si / ti) if ti > 0 else 0.0
            metrics[k] = (m1, m2, m1 * m2)
        return metrics

    met1 = calc_metrics_for_half(tot1, sdc1, sum_tot1)
    met2 = calc_metrics_for_half(tot2, sdc2, sum_tot2)

    # 5) 各半 Top-50%（按 score 降序），取交集
    n = len(keys)
    top_n = max(1, math.ceil(0.5 * n))

    def top_set(metrics):
        # metrics[k] = (m1, m2, score)
        ordered = sorted(metrics.items(), key=lambda kv: (-kv[1][2], kv[0][0], kv[0][1], kv[0][2]))
        return set(k for k, _ in ordered[:top_n])

    top1 = top_set(met1)
    top2 = top_set(met2)
    inter = top1 & top2

    if len(inter) < 2:
        return 0.0, 0.0, inter

    # 6) 在交集上计算 Spearman
    m1_half1 = [met1[k][0] for k in inter]
    m1_half2 = [met2[k][0] for k in inter]
    m2_half1 = [met1[k][1] for k in inter]
    m2_half2 = [met2[k][1] for k in inter]

    A = _spearman_corr(m1_half1, m1_half2)
    B = _spearman_corr(m2_half1, m2_half2)
    return A, B, inter


# -----------------------------
# 主流程（按你的新规则）
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze inst_exec.log and merge results with previous CSV; compute split-half Spearman A/B & manage _A snapshot."
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

    # ===== 新增：分半排序 & 前 50% 指令的 Spearman 相关（A/B）=====
    A, B, inter_keys = compute_A_B_with_random_split(effects_occ, results_occ)

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

    # 读取最近历史，计算“连续 >=0.9”的计数
    hist = _read_last_vals(info_path, k=10)  # 取最近最多 10 轮
    # 把当前一轮的 (A,B) 也视作已写入（hist 已含当前行）
    def streak_ge(vals, thr=0.9):
        cnt = 0
        for v in reversed(vals):
            if v >= thr:
                cnt += 1
            else:
                break
        return cnt

    As = [ab[0] for ab in hist]
    Bs = [ab[1] for ab in hist]
    A_streak = streak_ge(As, 0.9)
    B_streak = streak_ge(Bs, 0.9)

    # A 连续>=3 且 B 尚未连续>=3：另存 test_result_*_A.csv
    snap_A_path = _append_suffix_before_ext(out_path, "_A")
    if A_streak >= 3 and B_streak < 3:
        try:
            shutil.copyfile(out_path, snap_A_path)
            print(f"[Info] A has >=0.9 for {A_streak} consecutive rounds (B={B_streak}). Snapshot saved: {snap_A_path}")
        except Exception as e:
            print(f"[WARN] Failed to save snapshot {snap_A_path}: {e}")

    # 若 A 在曾经达到连续>=3 后又跌破（即当前连续计数 < 3），删除 _A 快照（若存在）
    if A_streak < 3:
        if os.path.exists(snap_A_path):
            try:
                os.remove(snap_A_path)
                print(f"[Info] A-streak dropped below 3. Snapshot removed: {snap_A_path}")
            except Exception as e:
                print(f"[WARN] Failed to remove {snap_A_path}: {e}")

    # 当 A 和 B 均连续 3 轮 >=0.9 时，上抛 exit99
    if A_streak >= 3 and B_streak >= 3:
        print(f"[EXIT] A and B have both been >=0.9 for 3 consecutive rounds (A_streak={A_streak}, B_streak={B_streak}). exit99.")
        sys.exit(99)

    # 正常结束
    return


if __name__ == "__main__":
    main()
