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
# 新增：仅新数据的 invalid 比例（保留原先定义）
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
# 工具：result_info 维护 & 快照
# -----------------------------

def _append_summary_csv(info_path: str, header_fields, row_dict: dict):
    """把一行摘要追加到 result_info（若文件不存在或表头不匹配则写表头）"""
    os.makedirs(os.path.dirname(info_path), exist_ok=True)

    write_header = False
    need_section_header = False
    if not os.path.exists(info_path) or os.path.getsize(info_path) == 0:
        write_header = True
    else:
        try:
            with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
            expected_first = ",".join(header_fields)
            if first_line != expected_first:
                need_section_header = True
        except Exception:
            need_section_header = True

    mode = "a" if os.path.exists(info_path) else "w"
    with open(info_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_fields)
        if write_header or need_section_header:
            w.writeheader()
        w.writerow(row_dict)


def _infer_next_cycle(info_path: str, header_fields):
    """读取 result_info，推断下一轮编号（同表头段内 +1）"""
    try:
        if not os.path.exists(info_path) or os.path.getsize(info_path) == 0:
            return 1
        header_line = ",".join(header_fields)
        with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f]
        last_idx = -1
        for i, ln in enumerate(lines):
            if ln.strip() == header_line:
                last_idx = i
        if last_idx >= 0:
            count = 0
            for ln in lines[last_idx + 1:]:
                if not ln.strip():
                    continue
                if ln.strip() == header_line:
                    count = 0
                    continue
                count += 1
            return count + 1 if count >= 0 else 1
        count_all = 0
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("time,"):
                continue
            count_all += 1
        return count_all + 1 if count_all >= 0 else 1
    except Exception:
        return 1


def _append_suffix_before_ext(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def _save_snapshot_if_first(out_csv_path: str, suffix: str):
    """仅当快照不存在时另存一次；存在时不覆盖。"""
    snap = _append_suffix_before_ext(out_csv_path, suffix)
    if not os.path.exists(snap):
        try:
            shutil.copyfile(out_csv_path, snap)
            print(f"Snapshot saved: {snap}")
        except Exception as e:
            print(f"WARN: failed to save snapshot {snap}: {e}")
    else:
        print(f"Snapshot already exists (kept): {snap}")
    return snap


# -----------------------------
# 新指标 + 停机/快照逻辑（按你的新要求）
# -----------------------------

def compute_metrics_and_maybe_stop_new(out_csv_path: str,
                                       coverage: float,
                                       info_path: str,
                                       perc_inv_new: float,
                                       snapshot_threshold_total: int = 9604,
                                       per_inst_threshold: int = 384):
    """
    新逻辑：
      - 读取 CSV，过滤 invalid_summary；
      - 指标1 = tot_inj / sum_noninvalid_totinj
      - 指标2 = SDC / tot_inj（tot_inj=0 时取 0）
      - score = 指标1 * 指标2，从大到小排序；
      - K = ceil(coverage * 非 invalid 指令数) 取 Top-K；
      - ratio = Top-K 内 tot_inj >= per_inst_threshold 的比例；
      - 每轮向 result_info 追加：cycle, perc_inv_new, noninvalid_tot_inj, coverage_ge384_ratio；
      - 若 ratio == 1 则打印信息并 exit(99)；
      - 若 sum_noninvalid_totinj >= 9604 且尚未留存，则另存 *_9604.csv。
    """
    # 读取 CSV
    rows = []
    if not os.path.exists(out_csv_path):
        print(f"CSV not found: {out_csv_path}")
        return None

    with open(out_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 非 invalid 行
    def _safe_int(s, d=0):
        try:
            return int(s)
        except Exception:
            return d

    noninv_rows = [r for r in rows if r.get("kernel", "") != "invalid_summary"]
    n_noninv = len(noninv_rows)

    sum_noninv_totinj = sum(_safe_int(r.get("tot_inj", 0)) for r in noninv_rows)

    # 分段快照（首次）：当非 invalid tot_inj 之和达到 384 / 1067 / 9604 时各自留存一次
    for thr in (384, 1067, snapshot_threshold_total):  # snapshot_threshold_total 默认就是 9604
        if sum_noninv_totinj >= thr:
            _save_snapshot_if_first(out_csv_path, f"_{thr}")

    # 计算指标与排序
    scores = []
    denom_total = max(sum_noninv_totinj, 0)
    for r in noninv_rows:
        tot = _safe_int(r.get("tot_inj", 0))
        sdc = _safe_int(r.get("SDC", 0))
        metric1 = (tot / denom_total) if denom_total > 0 else 0.0
        metric2 = (sdc / tot) if tot > 0 else 0.0
        score = metric1 * metric2
        scores.append({
            "kernel": r.get("kernel", ""),
            "inst_line": r.get("inst_line", ""),
            "inst_text": r.get("inst_text", ""),
            "tot_inj": tot,
            "SDC": sdc,
            "metric1": metric1,
            "metric2": metric2,
            "score": score,
        })

    scores.sort(key=lambda x: x["score"], reverse=True)

    # Top-K & 比例
    if n_noninv > 0:
        K = max(1, math.ceil(coverage * n_noninv))
        topK = scores[:K]
        cnt_ge = sum(1 for x in topK if x["tot_inj"] >= per_inst_threshold)
        ratio = cnt_ge / K
    else:
        K, topK, cnt_ge, ratio = 0, [], 0, 0.0

    # result_info 追加（加入 perc_inv_new）
    header_fields = ["cycle", "perc_inv_new", "noninvalid_tot_inj", "coverage_ge384_ratio"]
    cycle = _infer_next_cycle(info_path, header_fields)
    _append_summary_csv(
        info_path,
        header_fields,
        {
            "cycle": cycle,
            "perc_inv_new": f"{perc_inv_new:.6f}",
            "noninvalid_tot_inj": sum_noninv_totinj,
            "coverage_ge384_ratio": f"{ratio:.6f}",
        },
    )

    # 打印摘要
    print("===== Coverage Check (New Criterion) =====")
    print(f" Non-invalid instructions: {n_noninv}")
    print(f" Sum(non-invalid tot_inj): {sum_noninv_totinj}")
    print(f" Coverage (Top-K) K = {K} (coverage={coverage:.3f})")
    print(f" Ratio of Top-K with tot_inj >= {per_inst_threshold}: {ratio:.6f} ({cnt_ge}/{K})")
    print(f" perc_inv_new (this batch): {perc_inv_new:.6f}")

    # 若达到停机条件
    if K > 0 and cnt_ge == K:
        print("\nAll Top-K (by metric1*metric2) have tot_inj >= "
              f"{per_inst_threshold}. Triggering early stop (exit99).")
        print(" Top-K preview:")
        for i, x in enumerate(topK[:min(10, len(topK))], 1):
            print(f"  #{i} kernel={x['kernel']} line={x['inst_line']} tot_inj={x['tot_inj']} "
                  f"SDC={x['SDC']} m1={x['metric1']:.6f} m2={x['metric2']:.6f} score={x['score']:.6e}")
        sys.exit(99)

    # 未达标则返回摘要
    return {
        "cycle": cycle,
        "noninvalid_tot_inj": sum_noninv_totinj,
        "coverage_ge384_ratio": ratio,
        "K": K,
        "cnt_ge": cnt_ge,
        "perc_inv_new": perc_inv_new,
    }


# -----------------------------
# 主流程
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze inst_exec.log and merge results with previous CSV (new stop & snapshot rules)."
    )
    parser.add_argument("--app", "-a", required=True, help="Application name")
    parser.add_argument("--test", "-t", required=True, help="Test identifier", type=str)
    parser.add_argument("--component", "-c", required=True, help="Component set")
    parser.add_argument("--bitflip", "-b", required=True, help="Number of bit flips to inject")

    # 旧参数保留（但不使用），保证外部调用兼容
    parser.add_argument("--alpha", type=float, default=0.05, help="(ignored)")
    parser.add_argument("--eps_share", type=float, default=0.01, help="(ignored)")
    parser.add_argument("--eps_inv", type=float, default=0.01, help="(ignored)")
    parser.add_argument("--eps_sdc", type=float, default=0.02, help="(ignored)")
    parser.add_argument("--coverage", type=float, default=0.90, help="Top-K 覆盖率（用于新停机判据）")

    parser.add_argument("--batch_injections", type=int, default=0, help="(ignored)")

    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst_exec.log")

    # 解析本次日志
    effects_occ, results_occ, params_by_pair = parse_log(log_path)

    # 计算并保留 perc_inv_new
    perc_inv_new = _compute_perc_inv_from_new(effects_occ)

    # 写入/合并结果 CSV
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_result_{args.app}_{args.test}_{args.component}_{args.bitflip}.csv")

    out_path, total_masked, total_sdc, total_due, total_others, total_inj = write_csv(
        args.app, args.test, args.component, args.bitflip, effects_occ, results_occ, params_by_pair)

    # 记录 invalid 参数组合（保持原功能）
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

    # 对“invalid 但结果为 SDC”的新注入，拷贝 tmp.out 并记录信息（保持原功能）
    _collect_invalid_sdc_outputs(
        effects_occ,
        results_occ,
        params_by_pair,
        app=args.app,
        test=args.test,
        components=args.component,
        bitflip=args.bitflip,
    )

    # 新的停机/快照逻辑（加入 perc_inv_new）
    info_dir = "result_info"
    os.makedirs(info_dir, exist_ok=True)
    info_path = os.path.join(info_dir, f"result_info_{args.app}_{args.test}_{args.component}_{args.bitflip}.csv")

    summary = compute_metrics_and_maybe_stop_new(
        out_csv_path=out_path,
        coverage=args.coverage,
        info_path=info_path,
        perc_inv_new=perc_inv_new,
        snapshot_threshold_total=9604,
        per_inst_threshold=384,
    )

    if summary is not None:
        print("========== Final Summary ==========")
        print(f" Masked: {total_masked} | SDC: {total_sdc} | DUE: {total_due} | Others: {total_others}")
        print(f" Cycle={summary['cycle']} | NonInvalidTotInj={summary['noninvalid_tot_inj']} | "
              f"TopK>=384 Ratio={summary['coverage_ge384_ratio']:.6f} ({summary['cnt_ge']}/{summary['K']}) | "
              f"perc_inv_new={summary['perc_inv_new']:.6f}")


if __name__ == "__main__":
    main()
