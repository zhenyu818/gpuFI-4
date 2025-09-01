#!/usr/bin/env python3
import re
import sys
import os
import csv
from collections import defaultdict

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
    results = []
    
    for line in lines:
        if re.match(r"^[-]+$", line.strip()):
            continue
        if "Kernel" in line and "Instr_ID" in line:
            # Print header
            if not header_printed:
                print("Kernel,Instr_ID,Instruction,Masked,SDC,DUE,Eff_rate,SDC_rate,Total_Injections")
                header_printed = True
            continue
        if line.strip().startswith("Summary"):
            break

        parts = line.split("|")
        if len(parts) < 6:
            continue
        try:
            kernel = parts[0].strip()
            instr_id = parts[1].strip()
            instruction = parts[2].strip()
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

            results.append({
                'kernel': kernel,
                'instr_id': instr_id,
                'instruction': instruction,
                'masked': masked,
                'sdc': sdc,
                'due': due,
                'total': total,
                'eff_rate': eff_rate,
                'sdc_rate': sdc_rate
            })

            print(f"{kernel},{instr_id},{instruction},{masked},{sdc},{due},{eff_rate:.6f},{sdc_rate:.6f},{total}")
        except ValueError:
            continue
    
    return results, injection_sum

def load_existing_results(filename):
    """Load existing CSV results file"""
    if not os.path.exists(filename):
        return {}, 0
    
    results = {}
    total_injections = 0
    
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                kernel = row['Kernel']
                instr_id = row['Instr_ID']
                instruction = row['Instruction']
                masked = int(row['Masked'])
                sdc = int(row['SDC'])
                due = int(row['DUE'])
                
                key = (kernel, instr_id, instruction)
                if key not in results:
                    results[key] = {'masked': 0, 'sdc': 0, 'due': 0}
                
                results[key]['masked'] += masked
                results[key]['sdc'] += sdc
                results[key]['due'] += due
                total_injections += (masked + sdc + due)
                
            except (ValueError, KeyError):
                continue
    
    return results, total_injections

def merge_and_save_results(app_name, test_name, new_results, new_injection_sum):
    """Merge results and save to CSV file"""
    filename = f"test_result/test_result_{app_name}_{test_name}.csv"
    
    # Load existing results
    existing_results, existing_total = load_existing_results(filename)
    
    # Merge new results
    for result in new_results:
        key = (result['kernel'], result['instr_id'], result['instruction'])
        if key not in existing_results:
            existing_results[key] = {'masked': 0, 'sdc': 0, 'due': 0}
        
        existing_results[key]['masked'] += result['masked']
        existing_results[key]['sdc'] += result['sdc']
        existing_results[key]['due'] += result['due']
    
    # Calculate total injection count
    total_injections = existing_total + new_injection_sum
    
    # Ensure directory exists
    os.makedirs("test_result", exist_ok=True)
    
    # Write merged results to CSV
    with open(filename, "w", newline='', encoding="utf-8") as f:
        fieldnames = ['Kernel', 'Instr_ID', 'Instruction', 'Masked', 'SDC', 'DUE', 'Eff_rate', 'SDC_rate', 'Total_Injections']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Sort by kernel and instr_id
        sorted_keys = sorted(existing_results.keys(), key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else 0))
        
        for key in sorted_keys:
            kernel, instr_id, instruction = key
            data = existing_results[key]
            masked = data['masked']
            sdc = data['sdc']
            due = data['due']
            total = masked + sdc + due
            
            if total > 0:
                eff_rate = total / total_injections
                sdc_rate = sdc / total
            else:
                eff_rate = 0.0
                sdc_rate = 0.0
            
            writer.writerow({
                'Kernel': kernel,
                'Instr_ID': instr_id,
                'Instruction': instruction,
                'Masked': masked,
                'SDC': sdc,
                'DUE': due,
                'Eff_rate': f"{eff_rate:.6f}",
                'SDC_rate': f"{sdc_rate:.6f}",
                'Total_Injections': total
            })
    
    print(f"\nResults saved to: {filename}")
    print(f"Total injection count: {total_injections}")
    print(f"Number of merged instructions: {len(existing_results)}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analyze_fault.py <app_name> <test_name>")
        print("Example: python3 analyze_fault.py pathfinder 5-0")
        sys.exit(1)
    
    app_name = sys.argv[1]
    test_name = sys.argv[2]
    
    try:
        # Parse current test results
        new_results, new_injection_sum = parse_test_result("parse_exec.log")
        
        # Merge and save results
        merge_and_save_results(app_name, test_name, new_results, new_injection_sum)
        
    except FileNotFoundError:
        print("Error: parse_exec.log file not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
