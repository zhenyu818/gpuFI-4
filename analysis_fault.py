#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys


def normalize_result(s: str) -> str:
    x = s.strip().lower()
    if 'sdc' in x:
        return 'SDC'
    if 'due' in x:
        return 'DUE'
    if 'masked' in x:
        return 'Masked'
    return 'others'


def parse_log(log_path: str):
    # effects[(run_id, name)] -> {effective, kernel, inst_line, inst_text}
    effects = {}
    # results[(run_id, name)] -> category
    results = {}

    # Regex patterns (capture run id and tmp.out name)
    re_effects_start = re.compile(r"^\[Run\s+(\d+)\]\s+Effects from\s+(?:.+/)?(tmp\.out\d+):\s*$")
    re_writer = re.compile(r"^\[.*FI_WRITER\].*?->\s*(\S+)\s+PC=.*\(([^:()]+):(\d+)\)\s*(.*)$")
    re_effective = re.compile(r"^\[.*FI_EFFECTIVE\]")
    re_result = re.compile(r"^\[Run\s+(\d+)\]\s+(tmp\.out\d+):\s*(.*?)\s*$")

    current_key = None

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.rstrip('\n')

                m = re_effects_start.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    current_key = (run_id, name)
                    effects[current_key] = {
                        'effective': False,
                        'kernel': None,
                        'inst_line': None,
                        'inst_text': None,
                    }
                    continue

                if current_key is not None and re_effective.match(line):
                    effects[current_key]['effective'] = True
                    continue

                if current_key is not None:
                    m = re_writer.match(line)
                    if m:
                        kernel = m.group(1)
                        inst_line = int(m.group(3))
                        inst_text = m.group(4).strip()
                        rec = effects[current_key]
                        rec['kernel'] = kernel
                        rec['inst_line'] = inst_line
                        rec['inst_text'] = inst_text
                        continue

                m = re_result.match(line)
                if m:
                    run_id = int(m.group(1))
                    name = m.group(2)
                    res_text = m.group(3)
                    key = (run_id, name)
                    results[key] = normalize_result(res_text)
                    continue
    except FileNotFoundError:
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    # Build set of all runs from results only
    all_runs = set(results.keys())
    return effects, results, all_runs


def write_csv(app: str, test: str, effects, results, all_runs):
    out_dir = os.path.join('test_result')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_result_{app}_{test}.csv")

    invalid_counts = {'SDC': 0, 'DUE': 0, 'Masked': 0, 'others': 0}

    # Aggregate per-instruction counts for this run
    inst_counts = {}  # (kernel, inst_line, inst_text) -> {'Masked': n, 'SDC': n, 'DUE': n}

    def sort_key(run_key):
        run_id, name = run_key
        # Extract numeric id from tmp.outN
        nums = re.findall(r"\d+", name)
        n = int(nums[0]) if nums else 0
        return (run_id, n)

    for run_key in sorted(all_runs, key=sort_key):
        res_cat = results.get(run_key, 'others')

        eff_rec = effects.get(run_key)
        if eff_rec and eff_rec.get('effective'):
            kernel = eff_rec.get('kernel') or 'unknown'
            inst_line = eff_rec.get('inst_line') if eff_rec.get('inst_line') is not None else -1
            inst_text = eff_rec.get('inst_text') or 'unknown'
            key = (kernel, inst_line, inst_text)
            if key not in inst_counts:
                inst_counts[key] = {'Masked': 0, 'SDC': 0, 'DUE': 0}
            if res_cat in inst_counts[key]:
                inst_counts[key][res_cat] += 1
        else:
            # invalid injection
            if res_cat in invalid_counts:
                invalid_counts[res_cat] = invalid_counts.get(res_cat, 0) + 1

    # If an existing CSV exists, merge with it
    if os.path.exists(out_path):
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                has_invalid_cols = set(['invalid_Masked','invalid_SDC','invalid_DUE']).issubset(set(headers))
                for row in reader:
                    kernel = row.get('kernel', '')
                    if kernel == 'invalid_summary':
                        try:
                            if has_invalid_cols:
                                invalid_counts['Masked'] += int(row.get('invalid_Masked') or 0)
                                invalid_counts['SDC'] += int(row.get('invalid_SDC') or 0)
                                invalid_counts['DUE'] += int(row.get('invalid_DUE') or 0)
                            else:
                                invalid_counts['Masked'] += int(row.get('Masked') or 0)
                                invalid_counts['SDC'] += int(row.get('SDC') or 0)
                                invalid_counts['DUE'] += int(row.get('DUE') or 0)
                        except ValueError:
                            pass
                        continue

                    # Merge instruction rows
                    try:
                        inst_line_str = row.get('inst_line', '')
                        inst_line = int(inst_line_str) if str(inst_line_str).strip() != '' else -1
                    except ValueError:
                        inst_line = -1
                    inst_text = row.get('inst_text', '')
                    key = (kernel, inst_line, inst_text)
                    try:
                        m = int(row.get('Masked') or 0)
                        s = int(row.get('SDC') or 0)
                        d = int(row.get('DUE') or 0)
                    except ValueError:
                        m = s = d = 0
                    if key not in inst_counts:
                        inst_counts[key] = {'Masked': 0, 'SDC': 0, 'DUE': 0}
                    inst_counts[key]['Masked'] += m
                    inst_counts[key]['SDC'] += s
                    inst_counts[key]['DUE'] += d
        except Exception:
            # If any issue reading old CSV, fall back to not merging
            pass

    # Write CSV with merged per-instruction counts and a single invalid summary row (no extra invalid columns)
    fieldnames = [
        'kernel', 'inst_line', 'inst_text',
        'Masked', 'SDC', 'DUE',
        'tot_inj', 'valid_rate', 'SDC_rate'
    ]

    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by kernel then inst_line (numeric), then text
        for (kernel, inst_line, inst_text) in sorted(
            inst_counts.keys(), key=lambda k: (k[0], k[1], k[2])
        ):
            c = inst_counts[(kernel, inst_line, inst_text)]
            tot_inj = c.get('Masked', 0) + c.get('SDC', 0) + c.get('DUE', 0)
            # total injections across the project includes invalid; compute now
            total_project_inj = sum(v.get('Masked', 0) + v.get('SDC', 0) + v.get('DUE', 0) for v in inst_counts.values()) + \
                                invalid_counts.get('Masked', 0) + invalid_counts.get('SDC', 0) + invalid_counts.get('DUE', 0)
            valid_rate = (tot_inj / total_project_inj) if total_project_inj > 0 else 0.0
            sdc_rate = (c.get('SDC', 0) / tot_inj) if tot_inj > 0 else 0.0
            writer.writerow({
                'kernel': kernel,
                'inst_line': '' if inst_line is None or inst_line < 0 else inst_line,
                'inst_text': inst_text,
                'Masked': c.get('Masked', 0),
                'SDC': c.get('SDC', 0),
                'DUE': c.get('DUE', 0),
                'tot_inj': tot_inj,
                'valid_rate': f"{valid_rate:.6f}",
                'SDC_rate': f"{sdc_rate:.6f}",
            })

        # Append invalid summary row
        total_project_inj = sum(v.get('Masked', 0) + v.get('SDC', 0) + v.get('DUE', 0) for v in inst_counts.values()) + \
                            invalid_counts.get('Masked', 0) + invalid_counts.get('SDC', 0) + invalid_counts.get('DUE', 0)
        invalid_tot = invalid_counts.get('Masked', 0) + invalid_counts.get('SDC', 0) + invalid_counts.get('DUE', 0)
        invalid_valid_rate = (invalid_tot / total_project_inj) if total_project_inj > 0 else 0.0
        invalid_sdc_rate = (invalid_counts.get('SDC', 0) / invalid_tot) if invalid_tot > 0 else 0.0
        writer.writerow({
            'kernel': 'invalid_summary',
            'inst_line': '',
            'inst_text': '',
            'Masked': invalid_counts.get('Masked', 0),
            'SDC': invalid_counts.get('SDC', 0),
            'DUE': invalid_counts.get('DUE', 0),
            'tot_inj': invalid_tot,
            'valid_rate': f"{invalid_valid_rate:.6f}",
            'SDC_rate': f"{invalid_sdc_rate:.6f}",
        })

    # Compute merged totals for printing
    total_masked = sum(v.get('Masked', 0) for v in inst_counts.values()) + invalid_counts.get('Masked', 0)
    total_sdc = sum(v.get('SDC', 0) for v in inst_counts.values()) + invalid_counts.get('SDC', 0)
    total_due = sum(v.get('DUE', 0) for v in inst_counts.values()) + invalid_counts.get('DUE', 0)
    totals = {'Masked': total_masked, 'SDC': total_sdc, 'DUE': total_due}

    return out_path, invalid_counts, totals


def main():
    parser = argparse.ArgumentParser(description='Analyze inst_exec.log and summarize fault injection results.')
    parser.add_argument('--app', '-a', required=True, help='Application name')
    parser.add_argument('--test', '-t', required=True, help='Test identifier', type=str)
    args = parser.parse_args()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inst_exec.log')
    effects, results, all_runs = parse_log(log_path)

    out_path, invalid_counts, totals = write_csv(args.app, args.test, effects, results, all_runs)

    total_injections = totals['Masked'] + totals['SDC'] + totals['DUE']
    print(f"CSV written: {out_path}")
    print(f"Total injections: {total_injections}")
    print(f"Overall - Masked: {totals['Masked']}, SDC: {totals['SDC']}, DUE: {totals['DUE']}")


if __name__ == '__main__':
    main()
