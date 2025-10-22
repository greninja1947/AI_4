from starter import parse_bif

def check_probabilities(bif_data):
    complete, missing = 0, 0
    for v, info in bif_data.items():
        for key in info["probabilities"]:
            if -1 in info["probabilities"][key]:
                missing += 1
            else:
                complete += 1
            if abs(sum(info["probabilities"][key])-1)>=0.0001:
                print(f"[WARN] Probabilities for {v} with parents {key} do not sum to 1.")
                print(f"They sum to",sum(info["probabilities"][key]))
                return False
    # print(f"Variables with missing CPTs: {missing}")
    # print(f"Variables with full CPTs: {complete}\n")

    return True

def check_format_and_error():
    base = parse_bif("hailfinder.bif")
    solved = parse_bif("solved.bif")
    gold = parse_bif("gold_hailfinder.bif")

    print("\nChecking probability consistency...")
    if check_probabilities(solved):
        print("solved network has full CPTs for all variables.")
    else:
        print("solved network has missing CPT entries.")

    if check_probabilities(gold):
        print("gold network has full CPTs for all variables.")
    else:
        print("gold network has missing CPT entries.")


    print("\nChecking format consistency...")
    passed = True
    missing_vars = set(base.keys()) - set(solved.keys())
    extra_vars = set(solved.keys()) - set(base.keys())
    if missing_vars or extra_vars:
        print(f"Variable mismatch: missing {missing_vars}, extra {extra_vars}")
        passed =False
        return
    
    # Parent & value checks
    for v in base:
        if base[v]["parents"] != solved[v]["parents"]:
            passed =False
            print(f"Parent mismatch for {v}")
        if len(base[v]["values"]) != len(solved[v]["values"]):
            passed =False
            print(f"Value mismatch for {v}")

    # Unlearned values
    unlearned = [v for v in solved if any(-1 in row for row in solved[v]["probabilities"].values())]
    if unlearned:
        passed =False
        print(f"Variables with -1 (not learned): {unlearned}")

    if passed:
        print("Format checks passed.")
    # Compute total learning error
    print("\nComputing total learning error...")
    total_error = 0
    n=0
    for v in gold:
        for key, gold_row in gold[v]["probabilities"].items():
            if key in solved[v]["probabilities"]:
                sol_row = solved[v]["probabilities"][key]
                n+=len(sol_row)
                total_error += sum(abs(a - b) for a, b in zip(gold_row, sol_row))

    print(f"Total learning error: {(total_error/n):.6f}")


if __name__ == "__main__":
    check_format_and_error()
