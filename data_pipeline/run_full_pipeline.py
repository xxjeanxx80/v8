#!/usr/bin/env python3
"""
Script de chay day du pipeline: download raw data + them features
"""

import os
import sys
import subprocess

def run_script(script_name: str):
    """Chay mot Python script"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    if result.returncode != 0:
        print(f"\nError: {script_name} failed with exit code {result.returncode}")
        return False
    
    return True


def main():
    """Main function"""
    print("="*60)
    print("Full Data Pipeline: Download + Add Features")
    print("="*60)
    
    # Step 1: Download raw data
    if not run_script("download_raw_data.py"):
        print("\nPipeline stopped: Download failed")
        return
    
    # Step 2: Add features
    if not run_script("add_features.py"):
        print("\nPipeline stopped: Feature addition failed")
        return
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print("\nData files are ready in ../data/ directory")


if __name__ == "__main__":
    main()

