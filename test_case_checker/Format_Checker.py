from pathlib import Path
import sys
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import re
import sys
import time
from collections import defaultdict
from pprint import pprint
from itertools import product
import math

def parse_bif(filename):
    with open(filename, 'r') as f:
        text = f.read()
    
    var_pattern = re.compile(
        r'variable\s+(\w+)\s*\{\s*type\s+discrete\s*\[\s*\d+\s*\]\s*=\s*\{\s*([^\}]+)\s*\}\s*;', 
        re.MULTILINE
    )

    variables = {}
    for name, values_str in var_pattern.findall(text):
        values = [v.strip() for v in re.split(r'[,\s]+', values_str) if v.strip()]
        variables[name] = {
            "values": values,
            "parents": [],
            "probabilities": {}
        }
    prob_pattern = re.compile(
        r'probability\s*\(\s*([^\|\)]+?)\s*(?:\|\s*([^\)]+))?\s*\)\s*\{(.*?)\};',
        re.DOTALL | re.MULTILINE
    )

    for target, parents_str, body in prob_pattern.findall(text):
        target = target.strip()
        parents = [p.strip() for p in parents_str.split(',')] if parents_str else []

        if target not in variables:
            variables[target] = {"values": [], "parents": parents, "probabilities": {}}
        else:
            variables[target]["parents"] = parents

        # Clean body
        body = body.strip()
        if body.startswith("table"):
            # Simple unconditional case
            table_vals = re.findall(r'[-\d\.]+', body)
            probs = [float(v) if v != '-1' else -1 for v in table_vals]
            variables[target]["probabilities"][()] = probs
        else:
            # Conditional block with multiple parent value rows
            row_pattern = re.compile(r'\(([^\)]+)\)\s*([^\n;]+)')
            for parent_vals, probs_str in row_pattern.findall(body):
                key = tuple(v.strip() for v in parent_vals.split(','))
                probs = [float(x) if x != '-1' else -1 for x in re.findall(r'[-\d\.]+', probs_str)]
                variables[target]["probabilities"][key] = probs

    return variables


def _probability_warnings(bif_data) -> List[str]:
    warnings: List[str] = []
    for var, info in bif_data.items():
        for parents, probs in info["probabilities"].items():
            if -1 in probs:
                warnings.append(f"Missing probability entries for {var} with parents {parents}")
            else:
                # Only check sum if no missing values (-1)
                # Use > instead of >= to match old checker behavior
                total = sum(probs)
                if abs(total - 1.0) > 1e-4:
                    warnings.append(f"Probabilities for {var} with parents {parents} sum to {total:.6f}")
    return warnings


def _structure_warnings(base, solved) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    format_ok = True

    missing_vars = set(base.keys()) - set(solved.keys())
    extra_vars = set(solved.keys()) - set(base.keys())
    if missing_vars:
        format_ok = False
        warnings.append(f"Missing variables: {sorted(missing_vars)}")
    if extra_vars:
        format_ok = False
        warnings.append(f"Extra variables: {sorted(extra_vars)}")

    for var in base:
        if var not in solved:
            continue
        if base[var]["parents"] != solved[var]["parents"]:
            format_ok = False
            warnings.append(f"Parent mismatch for {var}")
        if base[var]["values"] != solved[var]["values"]:
            format_ok = False
            warnings.append(f"Value mismatch for {var}")
        if any(-1 in row for row in solved[var]["probabilities"].values()):
            format_ok = False
            warnings.append(f"Unlearned probabilities for {var}")

    return format_ok, warnings


def _total_error(solved, gold) -> Tuple[float, int]:
    total_error = 0.0
    n_entries = 0
    for var in gold:
        if var not in solved:
            continue
        for parents, gold_row in gold[var]["probabilities"].items():
            if parents not in solved[var]["probabilities"]:
                continue
            solved_row = solved[var]["probabilities"][parents]
            n_entries += len(solved_row)
            total_error += sum(abs(a - b) for a, b in zip(gold_row, solved_row))
    return total_error, n_entries


def check_format_and_error(base_path: str, solved_path: str, gold_path: str) -> None:
    base = parse_bif(base_path)
    solved = parse_bif(solved_path)
    gold = parse_bif(gold_path)

    warnings: List[str] = []

    prob_warnings = _probability_warnings(solved)
    warnings.extend(prob_warnings)

    format_ok, structural_warnings = _structure_warnings(base, solved)
    warnings.extend(structural_warnings)

    total_error, n_entries = _total_error(solved, gold)
    avg_te = (total_error / n_entries) if n_entries else float("inf")

    status = "PASSED" if format_ok else "FAILED"
    print(f"FORMAT_STATUS,{status}")
    print(f"TE_VALUE,{avg_te:.6f}")
    print(f"TOTAL_ERROR,{total_error:.6f}")
    print(f"ENTRY_COUNT,{n_entries}")

    for warning in warnings:
        print(f"WARNING,{warning}")


def main(args: List[str]) -> None:
    if len(args) != 3:
        print("Usage: python Format_Checker.py <base_bif> <solved_bif> <gold_bif>")
        sys.exit(1)

    check_format_and_error(args[0], args[1], args[2])


if __name__ == "__main__":
    main(sys.argv[1:])
