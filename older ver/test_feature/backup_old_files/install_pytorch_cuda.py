#!/usr/bin/env python3
"""
Script huong dan cai dat PyTorch voi CUDA support cho NVIDIA GPU
"""

import subprocess
import sys

def check_cuda():
    """Kiem tra CUDA version tren he thong"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Tim thay NVIDIA GPU!")
            print(result.stdout)
            return True
        else:
            print("Khong tim thay NVIDIA GPU hoac nvidia-smi khong co trong PATH")
            return False
    except FileNotFoundError:
        print("Khong tim thay nvidia-smi. Vui long cai dat NVIDIA Driver.")
        return False

def install_pytorch_cuda():
    """Cai dat PyTorch voi CUDA support"""
    print("="*60)
    print("CAI DAT PYTORCH VOI CUDA SUPPORT")
    print("="*60)
    
    # Kiem tra GPU
    if not check_cuda():
        print("\nCanh bao: Khong tim thay GPU. Ban co muon tiep tuc khong?")
        response = input("Nhan Enter de tiep tuc hoac Ctrl+C de huy: ")
    
    print("\nDang cai dat PyTorch voi CUDA 12.4...")
    print("CUDA 13.0 tuong thich nguoc voi CUDA 12.x")
    print("Neu van gap van de, vui long truy cap:")
    print("https://pytorch.org/get-started/locally/")
    
    # PyTorch voi CUDA 12.4 (tuong thich voi CUDA 13.0)
    # Khong can torchaudio cho project nay
    command = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ]
    
    print(f"\nChay lenh: {' '.join(command)}")
    print("\nDang cai dat... (co the mat mot chut thoi gian)")
    
    try:
        subprocess.check_call(command)
        print("\n" + "="*60)
        print("CAI DAT THANH CONG!")
        print("="*60)
        
        # Kiem tra lai
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("\nGPU da san sang su dung!")
        else:
            print("\nCanh bao: CUDA van chua available. Vui long kiem tra lai.")
        
    except subprocess.CalledProcessError as e:
        print(f"\nLoi khi cai dat: {e}")
        print("\nVui long thu cai dat thu cong:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except Exception as e:
        print(f"\nLoi: {e}")

if __name__ == "__main__":
    install_pytorch_cuda()

