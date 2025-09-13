#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys


def normalize_result(s: str) -> str:
    x = s.strip().lower()
    if "sdc" in x:
        return "SDC"
    if "due" in x:
        return "DUE"
    if "masked" in x:
        return "Masked"
    return "others"


def parse_log(log_path: str):
    """
    解析 inst_exec.log
    返回:
        effects: { (run_id, name): [ {kernel, inst_line, inst_text, src} ] }
        results: { (run_id, name): "Masked"/"SDC"/"DUE"/"others" }
        all_runs: set of keys
    """
    effects = {}
    results = {}

    re_effects_start = re.compile(
        r"^\[Run\s+(\d+)\]\s+Effects from\s+(?:.+/)?(tmp\.out\d+):\s*$"
    )
    re_writer = re.compile(
        r"^\[(?P<src>.*)FI_WRITER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    re_reader = re.compile(
        r"^\[(?P<src>.*)FI_READER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$"
    )
    re_result = re.compile(r"^\[Run\s+(\d+)\]\s+(tmp\.out\d+):\s*(.*?)\s*$")

    current_key = None
    writers = []
    readers = []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")

                # 新实验开始
                m = re_effects_start.match(line)
                if m:
                    # 保存上一个 run
                    if current_key is not None:
                        if writers:
                            effects[current_key] = writers
                        elif readers:
                            effects[current_key] = readers
                        else:
                            effects[current_key] = []

                    run_id = int(m.group(1))
                    name = m.group(2)
                    current_key = (run_id, name)
                    writers = []
                    readers = []
                    continue

                if current_key is not None:
                    # WRITER 优先
                    m = re_writer.match(line)
                    if m:
                        writers = [  # 一旦发现 WRITER，直接覆盖
                            {
                                "kernel": m.group(2),
                                "inst_line": int(m.group(4)),
                                "inst_text": m.group(5).strip(),
                                "src": m.group("src").rstrip("_"),
                            }
                        ]
                        continue

                    # READER（仅在没有 writer 时保留）
                    m = re_reader.match(line)
                    if m and not writers:
                        readers.append(
                            {
                                "kernel": m.group(2),
                                "inst_line": int(m.group(4)),
                                "inst_text": m.group(5).strip(),
                                "src": m.group("src").rstrip("_"),
                            }
                        )
                        continue

                # 结果分类
                m = re_result.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    res_text = m.group(3)
                    key = (run_id, name)
                    results[key] = normalize_result(res_text)
                    continue

        # 文件结束时，别忘了保存最后一个 run
        if current_key is not None:
            if writers:
                effects[current_key] = writers
            elif readers:
                effects[current_key] = readers
            else:
                effects[current_key] = []

    except FileNotFoundError:
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    all_runs = set(results.keys())
    return effects, results, all_runs


def write_csv(app: str, test: str, effects, results, all_runs):
    out_dir = os.path.join("test_result")
    os.makedirs(out_dir, exist_ok=True)
    # 移除 components 参数，只保留 app 和 test
    out_path = os.path.join(out_dir, f"test_result_{app}_{test}.csv")

    inst_counts = {}   # (kernel, inst_line, inst_text) -> {src -> {Masked,SDC,DUE}}
    all_srcs = set()

    def sort_key(run_key):
        run_id, name = run_key
        nums = re.findall(r"\d+", name)
        n = int(nums[0]) if nums else 0
        return (run_id, n)

    for run_key in sorted(all_runs, key=sort_key):
        res_cat = results.get(run_key, "others")
        eff_recs = effects.get(run_key, [])

        if eff_recs:
            for rec in eff_recs:
                kernel = rec.get("kernel") or "unknown"
                inst_line = rec.get("inst_line") if rec.get("inst_line") is not None else -1
                inst_text = rec.get("inst_text") or "unknown"
                src = rec.get("src", "unknown")
                key = (kernel, inst_line, inst_text)

                if key not in inst_counts:
                    inst_counts[key] = {}
                if src not in inst_counts[key]:
                    inst_counts[key][src] = {"Masked": 0, "SDC": 0, "DUE": 0}

                if res_cat in ["Masked", "SDC", "DUE"]:
                    inst_counts[key][src][res_cat] += 1
                    all_srcs.add(src)
        else:
            # invalid 注入
            kernel = "invalid_summary"
            inst_line = -1
            inst_text = ""
            key = (kernel, inst_line, inst_text)
            src = "invalid"
            if key not in inst_counts:
                inst_counts[key] = {}
            if src not in inst_counts[key]:
                inst_counts[key][src] = {"Masked": 0, "SDC": 0, "DUE": 0}
            if res_cat in ["Masked", "SDC", "DUE"]:
                inst_counts[key][src][res_cat] += 1
                all_srcs.add(src)

    # 构造 CSV 列名
    src_columns = []
    for src in sorted(all_srcs):
        src_columns += [f"{src}_Masked", f"{src}_SDC", f"{src}_DUE"]

    fieldnames = ["kernel", "inst_line", "inst_text"] + src_columns + [
        "Masked", "SDC", "DUE", "tot_inj"
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for kernel, inst_line, inst_text in sorted(inst_counts.keys(), key=lambda k: (k[0], k[1], k[2])):
            src_map = inst_counts[(kernel, inst_line, inst_text)]
            row = {
                "kernel": kernel,
                "inst_line": "" if inst_line < 0 else inst_line,
                "inst_text": inst_text,
            }

            tot_masked = tot_sdc = tot_due = 0
            for src in all_srcs:
                m = src_map.get(src, {}).get("Masked", 0)
                s = src_map.get(src, {}).get("SDC", 0)
                d = src_map.get(src, {}).get("DUE", 0)
                row[f"{src}_Masked"] = m
                row[f"{src}_SDC"] = s
                row[f"{src}_DUE"] = d
                tot_masked += m
                tot_sdc += s
                tot_due += d

            row["Masked"] = tot_masked
            row["SDC"] = tot_sdc
            row["DUE"] = tot_due
            row["tot_inj"] = tot_masked + tot_sdc + tot_due

            writer.writerow(row)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inst_exec.log and summarize fault injection results."
    )
    parser.add_argument("--app", "-a", required=True, help="Application name")
    parser.add_argument("--test", "-t", required=True, help="Test identifier", type=str)
    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst_exec.log")
    effects, results, all_runs = parse_log(log_path)

    out_path = write_csv(
        args.app, args.test, effects, results, all_runs
    )

    print(f"CSV written: {out_path}")


if __name__ == "__main__":
    main()
