#!/usr/bin/env python3
"""
Main pipeline - runs all analysis steps in order.

This orchestrates the entire process mining analysis:
1. analysis.py      - Profile and analyze the input CSVs
2. build_model.py   - Build the object-centric event model
3. kpis.py          - Compute KPIs for research questions
4. plots.py         - Generate visualizations

All output is generated to ./data/, ./model/, and ./kpi/ folders.
Existing outputs will be overwritten.

Usage:
    python main.py

Requirements:
    Install dependencies from requirements.txt:
    pip install -r requirements.txt
"""
import subprocess
import sys
from pathlib import Path


def run_step(step_num, script_name, description):
    """Run a Python script and report results."""
    print(f"\n{'=' * 70}")
    print(f"Step {step_num}: {description}")
    print(f"{'=' * 70}")
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Error: {script_name} not found at {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            check=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error: {description} failed: {e}")
        return False


def main():
    print("Process Mining Analysis Pipeline")
    print("=" * 70)
    
    steps = [
        (1, "analysis.py", "Data profiling and analysis"),
        (2, "build_model.py", "Build object-centric event model"),
        (3, "kpis.py", "Compute KPIs"),
        (4, "plots.py", "Generate visualizations"),
    ]
    
    results = []
    for step_num, script, description in steps:
        success = run_step(step_num, script, description)
        results.append((step_num, description, success))
    
    # Summary
    print(f"\n{'=' * 70}")
    print("Pipeline Summary")
    print("=" * 70)
    
    for step_num, description, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"Step {step_num}: {description:50s} {status}")
    
    all_passed = all(success for _, _, success in results)
    
    if all_passed:
        print(f"\n{'=' * 70}")
        print("All steps completed successfully!")
        print("Output folders:")
        print("  - data/     : Input and analyzed data")
        print("  - model/    : Generated object-centric event model")
        print("  - kpi/      : KPI reports and plots/")
        print(f"{'=' * 70}\n")
        return 0
    else:
        print(f"\n{'=' * 70}")
        print("⚠️  Pipeline completed with errors - see above for details")
        print(f"{'=' * 70}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
