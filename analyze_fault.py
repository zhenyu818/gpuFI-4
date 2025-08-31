#!/usr/bin/env python3
import re

def parse_test_result(filename):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Locate the summary section
    summary_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Summary (per kernel):"):
            summary_start = i
            break

    if summary_start is None:
        raise ValueError("Could not find the Summary section")

    # Compute injection_sum
    injection_sum = 0
    for line in lines[summary_start+2:]:
        if not line.strip() or line.startswith("---"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        try:
            masked = int(parts[1].strip())
            sdc = int(parts[2].strip())
            due = int(parts[3].strip())
            injection_sum += (masked + sdc + due)
        except ValueError:
            continue

    print(f"Total fault injections (injection_sum) = {injection_sum}\n")

    # Process each instruction line
    print("Results (with Eff_rate and SDC_rate):\n")
    header_printed = False
    for line in lines:
        if re.match(r"^[-]+$", line.strip()):
            continue
        if "Kernel" in line and "Instr_ID" in line:
            # Print header
            if not header_printed:
                print(line.strip() + " | Eff_rate | SDC_rate")
                header_printed = True
            continue
        if line.strip().startswith("Summary"):
            break

        parts = line.split("|")
        if len(parts) < 6:
            continue
        try:
            masked = int(parts[3].strip())
            sdc = int(parts[4].strip())
            due = int(parts[5].strip())
            total = masked + sdc + due
            if total > 0:
                eff_rate = total / injection_sum
                sdc_rate = sdc / total
            else:
                eff_rate = 0.0
                sdc_rate = 0.0

            print(f"{line.strip()} | {eff_rate:.6f} | {sdc_rate:.6f}")
        except ValueError:
            continue


if __name__ == "__main__":
    parse_test_result("test_result.txt")
