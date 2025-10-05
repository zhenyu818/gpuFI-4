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
from scipy.stats import norm

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
    # 2) 内联形式：同一行既有 Effects 前缀也有 Writer/Reader（新格式）
    re_effects_inline = re.compile(
        r"^\[Run\s+(\d+)\]\s+Effects from\s+(?:.+/)?(tmp\.out\d+):\s*(.*\S.*)$"
    )
    # 3) Writer/Reader 本体
    #    - 放宽 src 允许连字符；
    #    - 仍要求形如 "-> <kernel> PC=...(file:line) ..." 的结构（与你现有打印一致）
    re_writer = re.compile(
        r"^\[(?P<src>[-A-Za-z0-9_]+)_FI_WRITER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    re_reader = re.compile(
        r"^\[(?P<src>[-A-Za-z0-9_]+)_FI_READER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    # 4) 结果与参数
    re_result = re.compile(r"^\[Run\s+(\d+)\]\s+(tmp\.out\d+):\s*(.*?)\s*$")
    re_params = re.compile(r"^\[INJ_PARAMS\]\s+\[Run\s+(\d+)\]\s+(tmp\.out\d+)\s+(.*)$")

    latest_effects_by_pair = {}   # {(run_id,name): [records...]} 跨段缓存、去重后的最新汇总
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
        """把 existing 与 add 两份 record 列表合并去重"""
        if not existing:
            return _merge_unique(add, [])
        if not add:
            return _merge_unique(existing, [])
        return _merge_unique(existing + add, [])

    def flush_current_effects():
        """把当前累积的 cur_writers/readers 合并到 latest_effects_by_pair[cur_key]（去重/追加）"""
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

                # ---- (A) 先匹配“内联 Effects + Writer/Reader”的新格式 ----
                m = re_effects_inline.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    rest = m.group(3).strip()
                    new_key = (run_id, name)

                    # 如果切换到新 key，先把上一个 key 的累积刷进缓存
                    if cur_key != new_key:
                        flush_current_effects()
                        cur_key = new_key
                        cur_writers, cur_readers = [], []

                    # 尝试把“余下部分”作为 WRITER / READER 解析
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
                        # 不立即 flush，允许同一 (run_id,name) 的多条内联继续累积
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

                    # 如果 rest 不是 WRITER/READER，本行只是“起始+附带说明”，当作普通起始
                    # 此时不 flush，保持 cur_key，以便随后行继续累积
                    continue

                # ---- (B) 其次匹配旧格式的“独立 Effects 起始行” ----
                m = re_effects_start.match(line)
                if m:
                    # 每遇到新的 Effects 起始，先把上一个 key 的累积刷进缓存
                    flush_current_effects()
                    run_id = int(m.group(1))
                    name = m.group(2)
                    cur_key = (run_id, name)
                    cur_writers, cur_readers = [], []
                    continue

                # ---- (C) 在已有 cur_key 的上下文中累积 WRITER/READER（旧格式）----
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

                # ---- (D) INJ_PARAMS ----
                m = re_params.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    params_by_pair[(run_id, name)] = m.group(3).strip()
                    continue

                # ---- (E) 结果行：把结果与“该 pair 的全部已知 writer/reader”绑定 ----
                m = re_result.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    res = normalize_result(m.group(3))
                    pair = (run_id, name)

                    # 记录同一 pair 出现的第 idx 次结果
                    occ_counter[pair] += 1
                    idx = occ_counter[pair]
                    inj_key = (run_id, name, idx)

                    # 若当前正处于该 pair 的上下文，把“当前累积 + 历史缓存”一并合并
                    if cur_key == pair:
                        current_pack = _merge_unique(cur_writers, cur_readers)
                        existed = latest_effects_by_pair.get(pair, [])
                        recs = _merge_records(existed, current_pack)
                        latest_effects_by_pair[pair] = deepcopy(recs)
                    else:
                        # 否则退回到我们已缓存的该 pair 的最新合并结果
                        recs = latest_effects_by_pair.get(
                            pair,
                            [
                                {"kernel": "invalid_summary", "inst_line": -1, "inst_text": "", "src": "invalid"}
                            ],
                        )

                    effects_occ[inj_key] = deepcopy(recs)
                    results_occ[inj_key] = res
                    continue

        # 文件结束：把最后一个上下文刷入缓存
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
      - 同时复制 inst_exec.log 并重命名为 inst_exec_{app}_{test}_{components}_{bitflip}_r{run_id}_{name}.log
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
            # conservative fallback
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

        # destination name: include context to avoid collisions
        dest_name = f"{app}_{test}_{components}_{bitflip}_r{run_id}_{name}"
        dest_path = _ensure_unique_path(err_dir, dest_name)
        try:
            shutil.copyfile(src_path, dest_path)
            appended += 1
        except Exception:
            dest_path = "COPY_FAILED"

        # === 新增：复制 inst_exec.log 并重命名 ===
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
# 新指标与 CSV 读取
# -----------------------------

def _compute_perc_inv_from_new(effects_occ):
    """
    Perc_inv（仅新数据）：新数据中 invalid 的 tot_inj / 新数据中所有行的 tot_inj
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


# -----------------------------
# 新判据（coverage×384）与快照（9604）
# -----------------------------

def _read_csv_summary(path):
    """
    读取 CSV，返回：
      - rows_raw: 列表[dict]，按行原样（含 kernel/inst_text/tot_inj 等）
      - all_tot: 全部行 tot_inj 之和（含 invalid）
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


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _save_snapshot_if_first(out_csv_path: str, suffix: str = "_snapshot"):
    """
    仅当快照不存在时另存一次；存在时不覆盖。
    返回快照路径。
    """
    base, ext = os.path.splitext(out_csv_path)
    snap = f"{base}{suffix}{ext}"
    if not os.path.exists(snap):
        try:
            shutil.copyfile(out_csv_path, snap)
            print(f"Snapshot saved: {snap}")
        except Exception as e:
            print(f"WARN: failed to save snapshot {snap}: {e}")
    else:
        print(f"Snapshot already exists (kept): {snap}")
    return snap



def _save_noninvalid_sum_snapshot(out_csv_path: str, noninv_sum: int, threshold: int = 9604):
    """
    当“非 invalid 指令的 tot_inj 之和”达到阈值时，保存 test_result 的一次快照。
    规则：沿用原逻辑（首次达标才保存，不覆盖，之后不重复保存）。
    """
    if noninv_sum >= threshold:
        _save_snapshot_if_first(out_csv_path, f"_noninvalid_sum{threshold}")


def compute_coverage_and_maybe_stop(out_csv_path: str,
                                    coverage: float,
                                    info_path: str,
                                    snapshot_threshold: int = 9604,
                                    per_inst_target: int = 384,
                                    print_top: int = 10):
    """
    新判据：
      - 指标1 = 每条（非 invalid）tot_inj / 全部（非 invalid）tot_inj 之和
      - 指标2 = 每条（非 invalid）SDC / 该条（非 invalid）tot_inj
      - score = 指标1 * 指标2，按 score 由大到小排序
      - 令 K = ceil(coverage * 非invalid指令数)，取 Top-K
      - 若 Top-K 中“达到 per_inst_target(默认384)”的占比 == 1.0（即全部达标），则 exit(99)

    result_info 每轮保存：
      cycle, time, noninv_tot_inj, coverage_ge384_ratio

    返回：
      {"cycle": ..., "noninv_tot_inj": ..., "K": ..., "coverage_ge384_ratio": ..., "totals": {...}}
    """
    rows_raw, _ = _read_csv_summary(out_csv_path)

    # ------ 非 invalid 过滤与总注入数 ------
    noninv_rows = [r for r in rows_raw if r.get("kernel", "") != "invalid_summary"]
    noninv_sum = sum(_safe_int(r.get("tot_inj", 0)) for r in noninv_rows)

    # ------ 9604 阈值的 test_result 快照 ------
    _save_noninvalid_sum_snapshot(out_csv_path, noninv_sum, snapshot_threshold)

    # ------ 计算指标1*指标2 并排序 ------
    denom = noninv_sum if noninv_sum > 0 else 1
    items = []
    for r in noninv_rows:
        tot = _safe_int(r.get("tot_inj", 0))
        sdc = _safe_int(r.get("SDC", 0))
        m1 = (tot / denom) if denom > 0 else 0.0
        m2 = (sdc / tot) if tot > 0 else 0.0
        score = m1 * m2
        items.append({
            "kernel": r.get("kernel", ""),
            "inst_line": r.get("inst_line", ""),
            "inst_text": r.get("inst_text", ""),
            "tot_inj": tot,
            "SDC": sdc,
            "m1": m1,
            "m2": m2,
            "score": score
        })
    items.sort(key=lambda x: x["score"], reverse=True)

    # ------ 取覆盖集 Top-K ------
    n = len(items)
    K = max(1, math.ceil(coverage * n)) if n > 0 else 0
    topK = items[:K] if K > 0 else []

    # ------ 统计 Top-K 中 tot_inj ≥ 384 的占比 ------
    ge_cnt = sum(1 for it in topK if it["tot_inj"] >= per_inst_target)
    ratio = (ge_cnt / K) if K > 0 else 0.0

    # ------ 每轮汇总写入 result_info ------
    summary_fields = ["cycle", "time", "noninv_tot_inj", "coverage_ge384_ratio"]
    next_cycle = _infer_next_cycle(info_path, summary_fields)
    _append_summary_csv(info_path, summary_fields, {
        "cycle": next_cycle,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "noninv_tot_inj": str(noninv_sum),
        "coverage_ge384_ratio": f"{ratio:.4f}",
    })

    # ------ 打印关键进展与 Top-K 摘要 ------
    print(f"[Coverage] 非 invalid 指令数 = {n}，coverage = {coverage:.2%} → K = {K}")
    print(f"[Progress] 非 invalid tot_inj 之和 = {noninv_sum} "
          f"{'(已保存快照)' if noninv_sum >= snapshot_threshold else f'(未达 {snapshot_threshold}，待触发快照)'}")
    if K > 0:
        print(f"[Top-K] 已达标(≥{per_inst_target})：{ge_cnt}/{K} → 占比 {ratio:.2%}")
        header = f"{'rank':>4}  {'tot_inj':>7}  {'SDC':>5}  {'m1(tot/sum)':>13}  {'m2(SDC/tot)':>13}  {'score':>10}  kernel:line  inst_text"
        print(header)
        for i, it in enumerate(topK[:print_top], 1):
            print(f"{i:>4}  {it['tot_inj']:>7}  {it['SDC']:>5}  {it['m1']:.6f}      {it['m2']:.6f}      {it['score']:.6f}  "
                  f"{it['kernel']}:{it['inst_line']}  {it['inst_text']}")

    # ------ 汇总 totals 返回（与旧打印兼容）------
    tot_masked = sum(_safe_int(x.get("Masked", 0)) for x in rows_raw)
    tot_sdc    = sum(_safe_int(x.get("SDC", 0)) for x in rows_raw)
    tot_due    = sum(_safe_int(x.get("DUE", 0)) for x in rows_raw)
    tot_others = sum(_safe_int(x.get("Others", 0)) for x in rows_raw)
    ret = {
        "cycle": next_cycle,
        "noninv_tot_inj": noninv_sum,
        "K": K,
        "coverage_ge384_ratio": ratio,
        "totals": {"Masked": tot_masked, "SDC": tot_sdc, "DUE": tot_due, "Others": tot_others},
    }

    # ------ 早停（全部覆盖集均达到 384）------
    if K > 0 and ge_cnt == K:
        print("\nEARLY_STOP: 覆盖集(Top-K)内所有指令的 tot_inj 均 ≥ "
              f"{per_inst_target}。发送 exit99。")
        sys.exit(99)

    # 若你想改为“Top-K 中任意一条达 384 就停”，把上面的判断改为：
    # if K > 0 and ge_cnt > 0: sys.exit(99)

    return ret


# -----------------------------
# 主流程
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze inst_exec.log and merge results with previous CSV."
    )
    parser.add_argument("--app", "-a", required=True, help="Application name")
    parser.add_argument("--test", "-t", required=True, help="Test identifier", type=str)
    parser.add_argument("--component", "-c", required=True, help="Component set")
    parser.add_argument("--bitflip", "-b", required=True, help="Number of bit flips to inject")

    # 覆盖率
    parser.add_argument("--coverage", type=float, default=0.90, help="Top-K coverage fraction over non-invalid instructions (default 0.90)")

    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst_exec.log")

    # 解析本次日志
    effects_occ, results_occ, params_by_pair = parse_log(log_path)

    # Perc_inv（仅新数据）
    perc_inv_new = _compute_perc_inv_from_new(effects_occ)

    # 写入/合并结果 CSV
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_result_{args.app}_{args.test}_{args.component}_{args.bitflip}.csv")

    out_path, total_masked, total_sdc, total_due, total_others, total_inj = write_csv(
        args.app, args.test, args.component, args.bitflip, effects_occ, results_occ, params_by_pair)

    # 记录 invalid 参数组合
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

    # 对“invalid 但结果为 SDC”的新注入，拷贝 tmp.out 并记录信息
    _collect_invalid_sdc_outputs(
        effects_occ,
        results_occ,
        params_by_pair,
        app=args.app,
        test=args.test,
        components=args.component,
        bitflip=args.bitflip,
    )

    # 计算 Top-K 精度并决定是否早停 / 快照维护
    info_dir = "result_info"
    os.makedirs(info_dir, exist_ok=True)
    info_path = os.path.join(info_dir, f"result_info_{args.app}_{args.test}_{args.component}_{args.bitflip}.csv")

    summary = compute_coverage_and_maybe_stop(
        out_csv_path=out_path,
        coverage=args.coverage,
        info_path=info_path,
        snapshot_threshold=9604,   # 需求：非 invalid tot_inj 之和达到 9604 触发快照
        per_inst_target=384        # 需求：Top-K 内每条至少 384 次注入才算达标
    )

    if summary is not None:
        totals = summary.get("totals", {})
        print("========== Final Summary ==========")
        print(f" Masked: {totals.get('Masked', 0)} | SDC: {totals.get('SDC', 0)} | "
            f"DUE: {totals.get('DUE', 0)} | Others: {totals.get('Others', 0)}")
        print(
            f"Cycle={summary['cycle']} | noninvalid_tot_inj={summary['noninv_tot_inj']} | "
            f"coverage_K={summary['K']} | coverage_ge384_ratio={summary['coverage_ge384_ratio']:.4f}"
        )



if __name__ == "__main__":
    main()
