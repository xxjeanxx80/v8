#!/usr/bin/env python3
"""
Script de don dep cac file khong can thiet trong test_feature
"""

import os
import shutil

# Danh sach file co the xoa
files_to_remove = [
    # Setup/Install scripts (da cai xong)
    'install_pytorch_cuda.py',
    'install_pytorch_cuda_fixed.py',
    'check_and_install_deps.py',
    'activate_venv311.ps1',
    
    # Documentation (da setup xong)
    'GPU_SETUP_GUIDE.md',
    'README_GPU.md',
    
    # Test scripts cu (co phien ban moi)
    'ablation_test.py',  # Co ablation_test_improved.py
    'optimized_features.py',  # Khong con dung
    
    # Results cu (neu khong can so sanh)
    'ablation_results.csv',  # Co ablation_results_improved.csv
    'ablation_results.png',  # Hinh cu
    'correlation_heatmap.png',  # Hinh cu
    'feature_analysis_summary.csv',  # Ket qua cu
]

# Thu muc backup (neu muon backup truoc khi xoa)
backup_dir = 'backup_old_files'

def cleanup_files(backup=True):
    """Xoa cac file khong can thiet"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if backup:
        # Tao thu muc backup
        backup_path = os.path.join(base_dir, backup_dir)
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        print(f"Tao thu muc backup: {backup_path}")
    
    removed_count = 0
    not_found_count = 0
    
    print("\nDang don dep files...")
    print("="*60)
    
    for filename in files_to_remove:
        filepath = os.path.join(base_dir, filename)
        
        if os.path.exists(filepath):
            if backup:
                # Backup truoc khi xoa
                backup_filepath = os.path.join(backup_path, filename)
                shutil.copy2(filepath, backup_filepath)
                print(f"  Backup: {filename}")
            
            # Xoa file
            os.remove(filepath)
            print(f"  Da xoa: {filename}")
            removed_count += 1
        else:
            print(f"  Khong tim thay: {filename}")
            not_found_count += 1
    
    print("="*60)
    print(f"\nTong ket:")
    print(f"  - Da xoa: {removed_count} files")
    print(f"  - Khong tim thay: {not_found_count} files")
    
    if backup:
        print(f"\nBackup da luu vao: {backup_dir}/")
        print("Neu can khoi phuc, copy tu backup ve thu muc goc")

def main():
    """Main function"""
    print("="*60)
    print("CLEANUP SCRIPT - Don Dep File Khong Can Thiet")
    print("="*60)
    
    # Hoi xac nhan
    response = input("\nBan co muon backup truoc khi xoa? (y/n): ").strip().lower()
    backup = response == 'y'
    
    if backup:
        print("Se backup cac file truoc khi xoa.")
    else:
        print("Se xoa truc tiep (khong backup).")
    
    confirm = input("\nBan co chac muon xoa cac file nay? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        cleanup_files(backup=backup)
        print("\nHoan thanh don dep!")
    else:
        print("\nDa huy. Khong co file nao bi xoa.")

if __name__ == "__main__":
    main()

