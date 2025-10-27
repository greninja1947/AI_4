#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import re
import shutil
from pathlib import Path
from datetime import datetime

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'

def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔═══════════════════════════════════════════════════════════╗
║        BAYESIAN NETWORK TEST RUNNER v3.2                  ║
║        Weather Forecasting Assignment                     ║
╚═══════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")

def find_test_cases():
    """Find all test case directories."""
    tc_dir = Path("test_case_checker/tc")
    if not tc_dir.exists():
        print(f"{Colors.RED}Error: test_case_checker/tc directory not found!{Colors.ENDC}")
        return []
    
    test_cases = [d for d in tc_dir.iterdir() if d.is_dir()]
    return sorted(test_cases, key=lambda x: x.name)

def display_test_cases(test_cases):
    """Display available test cases with numbering."""
    print(f"{Colors.CYAN}{Colors.BOLD}Available Test Cases:{Colors.ENDC}\n")
    for i, tc in enumerate(test_cases, 1):
        # Check if test case has required files
        bif_file = list(tc.glob("*.bif"))
        bif_file = [f for f in bif_file if not f.name.startswith(('gold_', 'solved_'))]
        dat_file = list(tc.glob("*.dat"))
        gold_file = list(tc.glob("gold_*.bif"))
        
        status = f"{Colors.GREEN}✓{Colors.ENDC}" if (bif_file and dat_file and gold_file) else f"{Colors.RED}✗{Colors.ENDC}"
        
        print(f"  {Colors.BOLD}[{i}]{Colors.ENDC} {status} {Colors.CYAN}{tc.name}{Colors.ENDC}")
        if bif_file:
            print(f"      └─ BIF: {bif_file[0].name}")
        if dat_file:
            print(f"      └─ DAT: {dat_file[0].name}")
        if gold_file:
            print(f"      └─ GOLD: {gold_file[0].name}")
        print()

def select_test_cases(test_cases):
    """Allow user to select which test cases to run."""
    while True:
        choice = input(f"{Colors.YELLOW}Enter test case numbers (comma-separated) or 'all': {Colors.ENDC}").strip()
        
        if choice.lower() == 'all':
            return test_cases
        
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            selected = [test_cases[i-1] for i in indices if 1 <= i <= len(test_cases)]
            if selected:
                return selected
            else:
                print(f"{Colors.RED}Invalid selection. Try again.{Colors.ENDC}")
        except (ValueError, IndexError):
            print(f"{Colors.RED}Invalid input. Enter numbers separated by commas.{Colors.ENDC}")

def run_compile():
    """Run the compile script."""
    print(f"{Colors.YELLOW}Running compile.sh...{Colors.ENDC}")
    try:
        result = subprocess.run(['./compile.sh'], 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        if result.returncode == 0:
            print(f"{Colors.GREEN}✓ Compilation successful{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.RED}✗ Compilation failed{Colors.ENDC}")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"{Colors.RED}✗ Compilation error: {e}{Colors.ENDC}")
        return False

def get_python_command():
    """Determine which Python command is available."""
    for cmd in ['python3', 'python']:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, 
                                  timeout=5)
            if result.returncode == 0:
                return cmd
        except:
            continue
    return 'python3'

def run_test_case(tc_path, results):
    """Run a single test case and collect results."""
    tc_name = tc_path.name
    print_section(f"Running Test Case: {tc_name}")
    
    result = {
        'name': tc_name,
        'status': 'FAILED',
        'time': 0,
        'te_value': None,
        'total_error': None,
        'n': None,
        'format_passed': False,
        'error': None,
        'warning_count': 0
    }

    # Remove any previously generated solved networks left in the test case directory.
    for solved_extra in tc_path.glob("solved_*.bif"):
        try:
            solved_extra.unlink()
        except OSError:
            pass
    
    # Find required files
    bif_files = list(tc_path.glob("*.bif"))
    bif_files = [f for f in bif_files if not f.name.startswith(('gold_', 'solved_'))]
    dat_files = list(tc_path.glob("*.dat"))
    gold_files = list(tc_path.glob("gold_*.bif"))
    
    if not bif_files or not dat_files:
        result['error'] = "Missing required files (BIF or DAT)"
        print(f"{Colors.RED}✗ Missing required files{Colors.ENDC}")
        results.append(result)
        return
    
    if not gold_files:
        result['error'] = "Missing gold standard file"
        print(f"{Colors.RED}✗ Missing gold standard file{Colors.ENDC}")
        results.append(result)
        return
    
    bif_file = bif_files[0]
    dat_file = dat_files[0]
    gold_file = gold_files[0]
    
    print(f"{Colors.CYAN}BIF File: {bif_file.name}{Colors.ENDC}")
    print(f"{Colors.CYAN}DAT File: {dat_file.name}{Colors.ENDC}")
    print(f"{Colors.CYAN}GOLD File: {gold_file.name}{Colors.ENDC}\n")
    
    # Use absolute paths for the test files
    bif_path = bif_file.absolute()
    dat_path = dat_file.absolute()
    gold_path = gold_file.absolute()
    
    try:
        # Run the solution with actual file paths
        print(f"{Colors.YELLOW}Executing: ./run.sh {bif_path} {dat_path}{Colors.ENDC}")
        
        start_time = time.time()
        process = subprocess.Popen(
            ['./run.sh', str(bif_path), str(dat_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Live timer
        print(f"{Colors.DIM}Running", end='', flush=True)
        while process.poll() is None:
            print(".", end='', flush=True)
            time.sleep(0.5)
        print(f"{Colors.ENDC}")
        
        end_time = time.time()
        elapsed = end_time - start_time
        result['time'] = elapsed
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            result['error'] = f"Runtime error (exit code {process.returncode})"
            print(f"{Colors.RED}✗ Execution failed (exit code {process.returncode}){Colors.ENDC}")
            if stderr:
                print(f"{Colors.RED}STDERR: {stderr[:500]}{Colors.ENDC}")
        else:
            print(f"{Colors.GREEN}✓ Execution completed in {elapsed:.2f}s{Colors.ENDC}")

            solved_root = Path("solved_hailfinder.bif")
            if not solved_root.exists():
                result['error'] = "solved file not created"
                print(f"{Colors.RED}✗ Output file not created: {solved_root}{Colors.ENDC}")
            else:
                output_root = Path("test_case_checker/output")
                case_output = output_root / tc_name
                case_output.mkdir(parents=True, exist_ok=True)

                dest_solved = case_output / "solved_hailfinder.bif"
                if dest_solved.exists():
                    dest_solved.unlink()
                shutil.move(str(solved_root), dest_solved)

                print(f"\n{Colors.YELLOW}Running Format Checker...{Colors.ENDC}")
                python_cmd = get_python_command()

                checker_result = subprocess.run(
                    [python_cmd, 'test_case_checker/Format_Checker.py',
                     str(bif_path), str(dest_solved), str(gold_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                checker_output = checker_result.stdout.strip()
                if checker_result.stderr:
                    print(f"{Colors.YELLOW}Format checker stderr:{Colors.ENDC}\n{checker_result.stderr.strip()}")

                format_status = None
                warnings = []

                for line in checker_output.splitlines():
                    if not line:
                        continue
                    key, *rest = line.split(',', 1)
                    value = rest[0].strip() if rest else ''
                    if key == 'FORMAT_STATUS':
                        format_status = value
                    elif key == 'TE_VALUE':
                        try:
                            result['te_value'] = float(value)
                        except ValueError:
                            pass
                    elif key == 'TOTAL_ERROR':
                        try:
                            result['total_error'] = float(value)
                        except ValueError:
                            pass
                    elif key == 'ENTRY_COUNT':
                        try:
                            result['n'] = int(float(value))
                        except ValueError:
                            pass
                    elif key == 'WARNING':
                        warnings.append(value)

                if format_status:
                    color = Colors.GREEN if format_status == 'PASSED' else Colors.RED
                    print(f"{color}Format status: {format_status}{Colors.ENDC}")

                if result['te_value'] is not None:
                    print(f"{Colors.CYAN}TE Value: {result['te_value']:.6f}{Colors.ENDC}")
                if result['total_error'] is not None:
                    print(f"{Colors.CYAN}Total Error: {result['total_error']:.6f}{Colors.ENDC}")
                if result['n'] is not None:
                    print(f"{Colors.CYAN}Entries compared: {result['n']}{Colors.ENDC}")

                if format_status == 'PASSED':
                    result['format_passed'] = True

                if warnings:
                    result['warning_count'] = len(warnings)
                    print(f"{Colors.YELLOW}Format checker warnings: {len(warnings)} issue(s){Colors.ENDC}")

                if result['format_passed'] and result['te_value'] is not None:
                    result['status'] = 'PASSED'
    
    except subprocess.TimeoutExpired:
        result['error'] = "Execution timeout"
        print(f"{Colors.RED}✗ Timeout exceeded{Colors.ENDC}")
    except Exception as e:
        result['error'] = str(e)
        print(f"{Colors.RED}✗ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
    
    results.append(result)
    
    # Display result summary
    print(f"\n{Colors.BOLD}Result Summary:{Colors.ENDC}")
    print(f"  Status: {Colors.GREEN if result['status'] == 'PASSED' else Colors.RED}{result['status']}{Colors.ENDC}")
    print(f"  Time: {Colors.CYAN}{result['time']:.2f}s{Colors.ENDC}")
    if result['format_passed']:
        print(f"  Format: {Colors.GREEN}PASSED{Colors.ENDC}")
    else:
        print(f"  Format: {Colors.RED}FAILED{Colors.ENDC}")
    if result['te_value'] is not None:
        print(f"  TE Value: {Colors.CYAN}{result['te_value']:.6f}{Colors.ENDC}")
    if result['total_error'] is not None:
        print(f"  Total Error: {Colors.CYAN}{result['total_error']:.2f}{Colors.ENDC}")
    if result['n'] is not None:
        print(f"  N (entries): {Colors.CYAN}{result['n']}{Colors.ENDC}")
    if result['error']:
        print(f"  Error: {Colors.RED}{result['error']}{Colors.ENDC}")

def generate_report(results):
    """Generate result.log file."""
    print_section("Generating Report")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open('result.log', 'w') as f:
        f.write("="*70 + "\n")
        f.write("BAYESIAN NETWORK TEST RESULTS\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        total_passed = sum(1 for r in results if r['status'] == 'PASSED')
        total_tests = len(results)
        
        f.write(f"Summary: {total_passed}/{total_tests} tests passed\n")
        f.write("-"*70 + "\n\n")
        
        for result in results:
            f.write(f"Test Case: {result['name']}\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Execution Time: {result['time']:.2f}s\n")
            f.write(f"  Format Check: {'PASSED' if result['format_passed'] else 'FAILED'}\n")
            
            if result['te_value'] is not None:
                f.write(f"  TE Value: {result['te_value']:.6f}\n")
            if result['total_error'] is not None:
                f.write(f"  Total Error: {result['total_error']:.2f}\n")
            if result['n'] is not None:
                f.write(f"  N (entries): {result['n']}\n")
            if result['error']:
                f.write(f"  Error: {result['error']}\n")
            f.write("\n" + "-"*70 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write(f"  Total Tests: {total_tests}\n")
        f.write(f"  Passed: {total_passed}\n")
        f.write(f"  Failed: {total_tests - total_passed}\n")
        
        avg_time = sum(r['time'] for r in results) / len(results) if results else 0
        f.write(f"  Average Time: {avg_time:.2f}s\n")
        
        valid_te = [r['te_value'] for r in results if r['te_value'] is not None]
        if valid_te:
            avg_te = sum(valid_te) / len(valid_te)
            f.write(f"  Average TE: {avg_te:.6f}\n")
            f.write(f"  Min TE: {min(valid_te):.6f}\n")
            f.write(f"  Max TE: {max(valid_te):.6f}\n")
    
    print(f"{Colors.GREEN}✓ Report generated: result.log{Colors.ENDC}")
    
    # Display summary table
    print(f"\n{Colors.BOLD}{Colors.CYAN}Final Summary:{Colors.ENDC}\n")
    print(f"{'Test Case':<20} {'Status':<10} {'Time':<10} {'TE Value':<12} {'Format':<8}")
    print("-" * 70)
    for r in results:
        status_color = Colors.GREEN if r['status'] == 'PASSED' else Colors.RED
        status = f"{status_color}{r['status']}{Colors.ENDC}"
        time_str = f"{r['time']:.2f}s"
        te_str = f"{r['te_value']:.6f}" if r['te_value'] is not None else "N/A"
        format_str = "✓" if r['format_passed'] else "✗"
        print(f"{r['name']:<20} {status:<19} {time_str:<10} {te_str:<12} {format_str:<8}")
    
    print(f"\n{Colors.BOLD}Total: {total_passed}/{total_tests} passed{Colors.ENDC}")

def main():
    print_banner()
    
    # Check if we're in the right directory
    if not Path("compile.sh").exists() or not Path("run.sh").exists():
        print(f"{Colors.RED}Error: compile.sh or run.sh not found!{Colors.ENDC}")
        print(f"{Colors.YELLOW}Make sure you're running this from the main project directory.{Colors.ENDC}")
        return
    
    # Find test cases
    test_cases = find_test_cases()
    if not test_cases:
        print(f"{Colors.RED}No test cases found!{Colors.ENDC}")
        return
    
    # Display and select test cases
    display_test_cases(test_cases)
    selected = select_test_cases(test_cases)
    
    print(f"\n{Colors.GREEN}Selected {len(selected)} test case(s){Colors.ENDC}")
    
    # Confirm
    confirm = input(f"\n{Colors.YELLOW}Proceed with testing? (y/n): {Colors.ENDC}").strip().lower()
    if confirm != 'y':
        print(f"{Colors.YELLOW}Testing cancelled.{Colors.ENDC}")
        return
    
    # Compile
    print_section("Compilation")
    if not run_compile():
        print(f"{Colors.RED}Compilation failed. Exiting.{Colors.ENDC}")
        return
    
    # Run test cases
    results = []
    total_start = time.time()
    
    for i, tc in enumerate(selected, 1):
        print(f"\n{Colors.BOLD}{Colors.HEADER}[{i}/{len(selected)}]{Colors.ENDC}")
        run_test_case(tc, results)
    
    total_time = time.time() - total_start
    
    # Generate report
    generate_report(results)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}All tests completed in {total_time:.2f}s{Colors.ENDC}")
    print(f"{Colors.CYAN}Check result.log for detailed results.{Colors.ENDC}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Testing interrupted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)