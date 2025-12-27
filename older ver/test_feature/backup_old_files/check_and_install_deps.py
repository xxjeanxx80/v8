#!/usr/bin/env python3
"""
Script kiem tra va cai dat cac dependencies can thiet
"""

import sys
import subprocess

def check_and_install(package_name):
    """Kiem tra va cai dat package neu chua co"""
    try:
        __import__(package_name)
        print(f"  OK: {package_name} da duoc cai dat")
        return True
    except ImportError:
        print(f"  Dang cai dat {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
            print(f"  OK: {package_name} da duoc cai dat thanh cong")
            return True
        except subprocess.CalledProcessError:
            print(f"  LOI: Khong the cai dat {package_name}")
            return False

def main():
    """Kiem tra va cai dat tat ca dependencies"""
    print("="*60)
    print("KIEM TRA VA CAI DAT DEPENDENCIES")
    print("="*60)
    print(f"\nPython executable: {sys.executable}")
    print(f"Python version: {sys.version}\n")
    
    packages = [
        'numpy',
        'pandas',
        'torch',
        'sklearn',
        'matplotlib',
        'seaborn',
        'scipy'
    ]
    
    all_ok = True
    for package in packages:
        if not check_and_install(package):
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("TAT CA DEPENDENCIES DA SAN SANG!")
    else:
        print("CO LOI XAY RA, VUI LONG KIEM TRA LAI")
    print("="*60)

if __name__ == "__main__":
    main()

