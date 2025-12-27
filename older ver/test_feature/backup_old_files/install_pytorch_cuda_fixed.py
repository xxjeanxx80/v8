#!/usr/bin/env python3
"""
Script cai dat PyTorch voi CUDA support (fixed version)
Bo qua torchaudio vi khong can thiet cho project nay
"""

import subprocess
import sys

def install_pytorch_cuda():
    """Cai dat PyTorch voi CUDA support"""
    print("="*60)
    print("CAI DAT PYTORCH VOI CUDA SUPPORT")
    print("="*60)
    
    print("\nCUDA version tren may: 13.0")
    print("PyTorch se tuong thich nguoc voi CUDA 12.x")
    print("\nDang cai dat PyTorch voi CUDA 12.4...")
    print("(Bo qua torchaudio vi khong can thiet)")
    
    # Giai phap 1: Thu voi cu124
    commands = [
        # CUDA 12.4
        [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ],
        # Neu khong duoc, thu voi cu121
        [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ],
        # Fallback: CPU version (neu khong co CUDA support)
        [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision"
        ]
    ]
    
    for i, command in enumerate(commands, 1):
        print(f"\nThu phuong an {i}/{len(commands)}...")
        print(f"Chay lenh: {' '.join(command)}")
        
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Cai dat thanh cong!")
                
                # Kiem tra lai
                import torch
                print(f"\nPyTorch version: {torch.__version__}")
                print(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"CUDA version: {torch.version.cuda}")
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                    print("\nGPU da san sang su dung!")
                else:
                    print("\nCanh bao: CUDA van chua available.")
                    if i < len(commands):
                        print("Thu phuong an tiep theo...")
                        continue
                
                print("\n" + "="*60)
                print("HOAN TAT!")
                print("="*60)
                return True
            else:
                print(f"Loi: {result.stderr}")
                if i < len(commands):
                    print("Thu phuong an tiep theo...")
                    continue
                
        except Exception as e:
            print(f"Loi: {e}")
            if i < len(commands):
                print("Thu phuong an tiep theo...")
                continue
    
    print("\n" + "="*60)
    print("KHONG THE CAI DAT PYTORCH VOI CUDA")
    print("="*60)
    print("\nVui long cai dat thu cong:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    print("\nHoac truy cap: https://pytorch.org/get-started/locally/")
    return False

if __name__ == "__main__":
    install_pytorch_cuda()

