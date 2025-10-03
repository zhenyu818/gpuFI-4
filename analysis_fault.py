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
    对“invalid 且结果为 SDC”的新注入：复制 tmp.out，并追加记录到 error_classification/error_list.txt
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
# 统计学与结果落盘（Wilson CI + 快照）
# -----------------------------

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """
    Wilson 得分区间（双侧），返回 (lower, upper, half_width)
    """
    if n <= 0 or k < 0 or k > n:
        return 0.0, 1.0, 0.5
    z = norm.ppf(1 - alpha / 2.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    rad = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / (n * denom**2)) / denom  # 调整以提高精度
    lo = max(0.0, center - rad)
    hi = min(1.0, center + rad)
    return lo, hi, (hi - lo) / 2.0


def _append_summary_csv(info_path: str, header_fields, row_dict: dict):
    """
    将“本轮摘要变量”以单行形式追加到 result_info 对应的 CSV 中：
      - 若文件不存在：写入表头后再写入一行；
      - 若已存在：直接追加一行；若旧文件是旧格式（逐指令详情），则直接在其后追加（必要时附加一次新的表头）。
    """
    os.makedirs(os.path.dirname(info_path), exist_ok=True)

    write_header = False
    need_section_header = False
    if not os.path.exists(info_path) or os.path.getsize(info_path) == 0:
        write_header = True
    else:
        # 粗略检查首行是否为本摘要格式表头；若不是，则在末尾先写入一次新的表头
        try:
            with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
            expected_first = ",".join(header_fields)
            if first_line != expected_first:
                need_section_header = True
        except Exception:
            # 无法读取则退化为写表头
            need_section_header = True

    mode = "a" if os.path.exists(info_path) else "w"
    with open(info_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_fields)
        if write_header or need_section_header:
            w.writeheader()
        w.writerow(row_dict)


def _infer_next_cycle(info_path: str, header_fields):
    """
    基于 result_info 历史内容推断下一轮轮数：
      - 若文件不存在：返回 1；
      - 若存在，找到最后一次与当前表头完全相同的表头行，统计其后的数据行数 n，返回 n+1；
      - 若未找到匹配表头：返回 1。
    """
    try:
        if not os.path.exists(info_path) or os.path.getsize(info_path) == 0:
            return 1
        header_line = ",".join(header_fields)
        with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f]
        # 找到最后一个匹配的表头位置
        last_idx = -1
        for i, ln in enumerate(lines):
            if ln.strip() == header_line:
                last_idx = i
        if last_idx >= 0:
            # 统计其后非空且不等于表头的行数
            count = 0
            for ln in lines[last_idx + 1 : ]:
                if not ln.strip():
                    continue
                if ln.strip() == header_line:
                    # 不应出现，但防御
                    count = 0
                    continue
                count += 1
            return count + 1 if count >= 0 else 1
        # 回退策略：统计所有疑似数据行（排除以 time 开头的表头与空行）
        count_all = 0
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            # 忽略可能的旧表头
            if s.startswith("time,"):
                continue
            count_all += 1
        return count_all + 1 if count_all >= 0 else 1
    except Exception:
        return 1


def _estimate_required_n_for_halfwidth(p_tilde, z, eps):
    """
    近似所需样本量：n ≈ z^2 * p~ * (1-p~) / eps^2
    p~ 用 Wilson/Agresti–Coull 的中心值以稳健化 0/1 边界。
    """
    p_tilde = min(max(p_tilde, 1e-12), 1 - 1e-12)
    return math.ceil((z * z) * p_tilde * (1.0 - p_tilde) / (eps * eps))


def _append_suffix_before_ext(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def _ensure_removed(path: str) -> bool:
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception as e:
        print(f"WARN: failed to remove {path}: {e}")
    return False


def _save_snapshot_if_first(out_csv_path: str, suffix: str):
    """
    仅当快照不存在时另存一次；存在时不覆盖。
    """
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


def compute_metrics_and_maybe_stop(out_csv_path: str,
                                   alpha: float,
                                   eps_share: float,
                                   eps_inv: float,
                                   eps_sdc: float,
                                   coverage: float,
                                   info_path: str,
                                   perc_inv_new: float,
                                   batch_injections: int = 0):
    """
    覆盖率驱动的 Top-K 判据：
      - 排除 invalid_summary；
      - score = share * sdc_rate；
      - K = ceil(coverage * 非invalid数量)；
      - 若 Top-K 中每条都满足：share_half <= eps_share 且 sdc_half <= eps_sdc，则 exit(99)。
    快照策略：
      - 仅 share 达标（sdc 未达标）：若无 *_share_exited.csv 则另存；若之后 share 失效则删除；
      - 仅 sdc 达标（share 未达标）：同理维护 *_sdc_exited.csv；
      - 两者同达标：不另存、不改名，直接 exit(99)。
    """
    rows_raw, all_tot = _read_csv_summary(out_csv_path)
    z = norm.ppf(1 - alpha / 2.0)

    # 计算 invalid 与非 invalid 的总注入数
    invalid_tot = 0
    for r in rows_raw:
        try:
            if r.get("kernel", "") == "invalid_summary":
                invalid_tot += _safe_int(r.get("tot_inj", 0))
        except Exception:
            pass
    noninvalid_tot = max(0, all_tot - invalid_tot)

    # 指标1（全局）：invalid 占比与半宽（采用 Wilson）
    if all_tot > 0:
        inv_share = invalid_tot / all_tot
        _, _, inv_half = wilson_ci(invalid_tot, all_tot, alpha)
    else:
        inv_share, inv_half = 0.0, 0.5

    # 逐行指标（Top-K 排序/判定只考虑非 invalid_summary；分母改为 noninvalid_tot）
    rows_metrics = []
    for r in rows_raw:
        kernel = r.get("kernel", "")
        inst_line = r.get("inst_line", "")
        inst_text = r.get("inst_text", "")
        tot_inj = _safe_int(r.get("tot_inj", 0))
        masked = _safe_int(r.get("Masked", 0))
        sdc = _safe_int(r.get("SDC", 0))
        due = _safe_int(r.get("DUE", 0))
        others = _safe_int(r.get("Others", 0))

        # 指标2（share，排除 invalid）：把“该指令被抽中一次”视作一次“成功”，试验数为 noninvalid_tot
        if kernel != "invalid_summary" and noninvalid_tot > 0:
            share = tot_inj / noninvalid_tot
            lo_s, hi_s, half_s = wilson_ci(tot_inj, noninvalid_tot, alpha)
        else:
            share = 0.0
            lo_s, hi_s, half_s = 0.0, 1.0, 0.5

        # 指标3（SDC rate）：对 tot=0 的行，CI 不可判定（记为 None）
        if tot_inj > 0:
            sdc_rate = sdc / tot_inj
            lo_p, hi_p, half_p = wilson_ci(sdc, tot_inj, alpha)
        else:
            sdc_rate = 0.0
            lo_p, hi_p, half_p = None, None, None

        score = share * (sdc_rate if sdc_rate is not None else 0.0)

        rows_metrics.append({
            "kernel": kernel,
            "inst_line": inst_line,
            "inst_text": inst_text,
            "Masked": masked, "SDC": sdc, "DUE": due, "Others": others, "tot_inj": tot_inj,
            "share": share, "share_ci": (lo_s, hi_s), "share_ci_half": half_s,
            "sdc_rate": sdc_rate, "sdc_ci": (lo_p, hi_p) if lo_p is not None else None,
            "sdc_ci_half": half_p, "score": score
        })

    # 不再写出逐指令的 result_info；改为每轮写入一条摘要

    # 仅对非 invalid_summary 的指令排序并取 Top-K（由 coverage 唯一决定）
    candidates = [r for r in rows_metrics if r["kernel"] != "invalid_summary"]
    if len(candidates) == 0:
        print("No non-invalid rows yet; cannot evaluate stopping.")
        # 仍然写入一条摘要（Top-K 不可评估时，达标比例置为 0）
        summary_fields = [
            "cycle", "time", "perc_inv_new", "inv_half", "eps_inv",
            "share_pass_ratio", "sdc_pass_ratio"
        ]
        # 轮数：基于历史数据推断
        next_cycle = _infer_next_cycle(info_path, summary_fields)
        summary_row = {
            "cycle": next_cycle,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "perc_inv_new": f"{perc_inv_new:.6f}",
            "inv_half": f"{inv_half:.6f}",
            "eps_inv": f"{eps_inv:.4f}",
            "share_pass_ratio": f"{0.0:.4f}",
            "sdc_pass_ratio": f"{0.0:.4f}",
        }

        _append_summary_csv(info_path, summary_fields, summary_row)
        # 汇总总计（保留分类计数）
        tot_masked = sum(_safe_int(x.get("Masked", 0)) for x in rows_raw)
        tot_sdc    = sum(_safe_int(x.get("SDC", 0)) for x in rows_raw)
        tot_due    = sum(_safe_int(x.get("DUE", 0)) for x in rows_raw)
        tot_others = sum(_safe_int(x.get("Others", 0)) for x in rows_raw)
        return {
            "cycle": summary_row["cycle"],
            "perc_inv_new": float(summary_row["perc_inv_new"]),
            "inv_half": float(summary_row["inv_half"]),
            "eps_inv": float(summary_row["eps_inv"]),
            "share_pass_ratio": float(summary_row["share_pass_ratio"]),
            "sdc_pass_ratio": float(summary_row["sdc_pass_ratio"]),
            "totals": {"Masked": tot_masked, "SDC": tot_sdc, "DUE": tot_due, "Others": tot_others},
        }

    K = max(1, math.ceil(coverage * len(candidates)))
    candidates.sort(key=lambda r: r["score"], reverse=True)
    topK = candidates[:K]

    # Top-K 上的判定
    share_pass = all(r["share_ci_half"] is not None and r["share_ci_half"] <= eps_share for r in topK)
    sdc_pass   = all(r["sdc_ci_half"]   is not None and r["sdc_ci_half"]   <= eps_sdc   for r in topK)
    share_pass_cnt = sum(1 for r in topK if r["share_ci_half"] is not None and r["share_ci_half"] <= eps_share)
    sdc_pass_cnt   = sum(1 for r in topK if r["sdc_ci_half"]   is not None and r["sdc_ci_half"]   <= eps_sdc)
    share_pass_ratio = (share_pass_cnt / K) if K > 0 else 0.0
    sdc_pass_ratio   = (sdc_pass_cnt / K) if K > 0 else 0.0

    # 失败项统计与注入量估计（仅总结性输出，不做逐项枚举）
    bad_share = [r for r in topK if not (r["share_ci_half"] is not None and r["share_ci_half"] <= eps_share)]
    bad_sdc   = [r for r in topK if not (r["sdc_ci_half"]   is not None and r["sdc_ci_half"]   <= eps_sdc)]

    if bad_share:
        if noninvalid_tot > 0:
            req_totals = []
            for r in bad_share:
                p_tilde = (r["tot_inj"] + z * z / 2.0) / (noninvalid_tot + z * z)
                n_req = _estimate_required_n_for_halfwidth(p_tilde, z, eps_share)
                req_totals.append(n_req)
            add_total_needed = max(req_totals) - noninvalid_tot if req_totals else 0
            add_total_needed = max(0, add_total_needed)
        else:
            add_total_needed = 0
        print(f" SHARE not met on Top-K: {len(bad_share)} items. "
              f"Approx additional TOTAL injections needed: ~{add_total_needed}")
        if batch_injections and batch_injections > 0:
            batches = math.ceil(add_total_needed / batch_injections) if add_total_needed > 0 else 0
            print(f"   (If each batch ~{batch_injections}, need ~{batches} more batches for share)")
    else:
        print(" SHARE precision OK on Top-K.")

    if bad_sdc:
        # 估算每条指令各自需要的样本量；这里只汇总最大缺口
        max_add_i = 0
        for r in bad_sdc:
            if r["tot_inj"] > 0:
                p_tilde_sdc = (r["SDC"] + z * z / 2.0) / (r["tot_inj"] + z * z)
                n_req_i = _estimate_required_n_for_halfwidth(p_tilde_sdc, z, eps_sdc)
                add_i = max(0, n_req_i - r["tot_inj"])
                if add_i > max_add_i:
                    max_add_i = add_i
            else:
                max_add_i = max(max_add_i, 1)  # tot=0 时至少需要>0的注入
        print(f" SDC not met on Top-K: {len(bad_sdc)} items. "
              f"Max per-instruction additional injections needed: +{max_add_i}")
    else:
        print(" SDC precision OK on Top-K.")

    # 追加“单行摘要”到 result_info CSV，并在控制台打印一行摘要（中文）
    summary_fields = [
        "cycle", "time", "perc_inv_new", "inv_half", "eps_inv",
        "share_pass_ratio", "sdc_pass_ratio"
    ]
    next_cycle = _infer_next_cycle(info_path, summary_fields)
    summary_row = {
        "cycle": next_cycle,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "perc_inv_new": f"{perc_inv_new:.6f}",
        "inv_half": f"{inv_half:.6f}",
        "eps_inv": f"{eps_inv:.4f}",
        "share_pass_ratio": f"{share_pass_ratio:.4f}",
        "sdc_pass_ratio": f"{sdc_pass_ratio:.4f}",
    }

    _append_summary_csv(info_path, summary_fields, summary_row)

    # -----------------
    # 快照维护（只保存“第一次达标”的快照；变为不达标则删除）
    # -----------------
    share_snap = _append_suffix_before_ext(out_csv_path, "_share_exited")
    sdc_snap   = _append_suffix_before_ext(out_csv_path, "_sdc_exited")
    inv_snap   = _append_suffix_before_ext(out_csv_path, "_inv")

    if share_pass and not sdc_pass:
        _save_snapshot_if_first(out_csv_path, "_share_exited")
        if os.path.exists(sdc_snap):
            if _ensure_removed(sdc_snap):
                print(f"Snapshot removed (sdc fell back): {sdc_snap}")

    elif sdc_pass and not share_pass:
        _save_snapshot_if_first(out_csv_path, "_sdc_exited")
        if os.path.exists(share_snap):
            if _ensure_removed(share_snap):
                print(f"Snapshot removed (share fell back): {share_snap}")

    elif not share_pass and not sdc_pass:
        if os.path.exists(share_snap):
            if _ensure_removed(share_snap):
                print(f"Snapshot removed (share fell back): {share_snap}")
        if os.path.exists(sdc_snap):
            if _ensure_removed(sdc_snap):
                print(f"Snapshot removed (sdc fell back): {sdc_snap}")

    else:
        # share_pass and sdc_pass 同时为 True：不另存也不改名；直接 exit(99)
        pass

    # 指标1（invalid 占比）单独维护快照：符合精度则另存 *_inv.csv；不符合则移除
    inv_pass = (inv_half <= eps_inv)
    if inv_pass:
        _save_snapshot_if_first(out_csv_path, "_inv")
    else:
        if os.path.exists(inv_snap):
            if _ensure_removed(inv_snap):
                print(f"Snapshot removed (inv fell back): {inv_snap}")

    # 早停：两者同时达标（Top-K 上）
    # 组织返回给 main 的摘要与总计（以便最后统一中文打印）
    tot_masked = sum(_safe_int(x.get("Masked", 0)) for x in rows_raw)
    tot_sdc    = sum(_safe_int(x.get("SDC", 0)) for x in rows_raw)
    tot_due    = sum(_safe_int(x.get("DUE", 0)) for x in rows_raw)
    tot_others = sum(_safe_int(x.get("Others", 0)) for x in rows_raw)
    ret = {
        "cycle": summary_row["cycle"],
        "perc_inv_new": float(summary_row["perc_inv_new"]),
        "inv_half": float(summary_row["inv_half"]),
        "eps_inv": float(summary_row["eps_inv"]),
        "share_pass_ratio": float(summary_row["share_pass_ratio"]),
        "sdc_pass_ratio": float(summary_row["sdc_pass_ratio"]),
        "totals": {"Masked": tot_masked, "SDC": tot_sdc, "DUE": tot_due, "Others": tot_others},
    }

    if share_pass and sdc_pass:
        print("\nEARLY_STOP: Both SHARE and SDC precision targets are met on Top-K.")
        print("Returning exit99 (no renaming; snapshots unchanged).")
        sys.exit(99)

    # 未达标：继续下一批
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

    # 置信区间与阈值/覆盖率
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level (default: 0.05 => 95% CI)")
    parser.add_argument("--eps_share", type=float, default=0.01, help="Max half-width for per-instruction share (default 0.01)")
    parser.add_argument("--eps_inv", type=float, default=0.01, help="Max half-width for invalid share (default 0.01)")
    parser.add_argument("--eps_sdc", type=float, default=0.02, help="Max half-width for per-instruction SDC rate (default 0.02)")
    parser.add_argument("--coverage", type=float, default=0.90, help="Top-K coverage fraction over non-invalid instructions (default 0.90)")

    # 打印辅助（不影响判据；可为 0）
    parser.add_argument("--batch_injections", type=int, default=0, help="(Optional) Approx injections per batch for planning printouts")

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

    # 控制台输出
    # 先不打印传统摘要，由 compute_metrics_and_maybe_stop 的返回统一打印

    # 计算 Top-K 精度并决定是否早停 / 快照维护
    info_dir = "result_info"
    os.makedirs(info_dir, exist_ok=True)
    info_path = os.path.join(info_dir, f"result_info_{args.app}_{args.test}_{args.component}_{args.bitflip}.csv")

    summary = compute_metrics_and_maybe_stop(
        out_csv_path=out_path,
        alpha=args.alpha,
        eps_share=args.eps_share,
        eps_inv=args.eps_inv,
        eps_sdc=args.eps_sdc,
        coverage=args.coverage,
        info_path=info_path,
        perc_inv_new=perc_inv_new,
        batch_injections=args.batch_injections
    )

    if summary is not None:
        totals = summary.get("totals", {})
        print("========== Final Summary ==========")
        print(f" Masked: {totals.get('Masked', total_masked)} | SDC: {totals.get('SDC', total_sdc)} | DUE: {totals.get('DUE', total_due)} | Others: {totals.get('Others', total_others)}")
        print(
            f"Cycle={summary['cycle']} | perc_inv_new={summary['perc_inv_new']:.6f} | "
            f"Metric1 Half-width={summary['inv_half']:.6f} | Metric1 Target Accuracy={summary['eps_inv']:.4f} | "
            f"Metric2 Pass Ratio={summary['share_pass_ratio']:.4f} | Metric3 Pass Ratio={summary['sdc_pass_ratio']:.4f}"
        )



if __name__ == "__main__":
    main()
