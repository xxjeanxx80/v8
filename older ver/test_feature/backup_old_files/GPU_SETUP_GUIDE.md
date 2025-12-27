# Hướng dẫn cài đặt PyTorch với CUDA cho GPU

## Vấn đề hiện tại

- Python version: 3.14.0
- CUDA trên máy: 13.0
- PyTorch hiện tại: 2.9.1+cpu (chỉ có CPU build)

## Giải pháp

PyTorch 2.9.1 cho Python 3.14 có thể chỉ có CPU build. Có các lựa chọn sau:

### Lựa chọn 1: Sử dụng Nightly Build (Khuyến nghị)

Nightly build thường có CUDA support sớm hơn:

```powershell
# Trong virtual environment
pip uninstall torch torchvision
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
```

### Lựa chọn 2: Downgrade Python về 3.12

Python 3.12 được PyTorch hỗ trợ đầy đủ:

```powershell
# Tạo virtual environment mới với Python 3.12
python3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Lựa chọn 3: Sử dụng CPU (Chậm hơn nhưng vẫn chạy được)

Code đã được cập nhật để tự động fallback về CPU nếu không có GPU.

## Kiểm tra sau khi cài đặt

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Lưu ý

- Code đã được cập nhật để tự động sử dụng GPU nếu có
- Nếu không có GPU, sẽ tự động fallback về CPU
- Training với CPU sẽ chậm hơn nhưng vẫn chạy được

