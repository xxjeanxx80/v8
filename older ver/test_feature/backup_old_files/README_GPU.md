# Hướng dẫn sử dụng GPU với Python 3.11

## Đã cài đặt thành công!

- **Python version**: 3.11.9
- **PyTorch version**: 2.5.1+cu121
- **CUDA available**: True
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
- **GPU Memory**: 4.0 GB

## Cách sử dụng

### 1. Kích hoạt virtual environment Python 3.11

Trong PowerShell, chạy:

```powershell
cd "C:\Users\xxjea\Downloads\DSS DATA"
.venv311\Scripts\Activate.ps1
```

Hoặc chạy script tự động:

```powershell
cd test_feature
.\activate_venv311.ps1
```

### 2. Chạy các test

Sau khi kích hoạt virtual environment, bạn có thể chạy:

```powershell
# Chạy tất cả tests
python comprehensive_test_runner.py

# Hoặc chạy từng test riêng
python threshold_optimization.py
python feature_combination_test.py
python advanced_ablation_test.py
python trading_strategy_evaluator.py
```

### 3. Kiểm tra GPU hoạt động

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Lưu ý

- Virtual environment Python 3.11 được lưu tại: `.venv311/`
- Virtual environment Python 3.14 cũ vẫn ở: `.venv/` (không có GPU support)
- Code sẽ tự động sử dụng GPU khi có sẵn
- Training với GPU sẽ nhanh hơn nhiều so với CPU

## So sánh hiệu suất

- **CPU**: ~10-30 phút cho mỗi test
- **GPU**: ~1-5 phút cho mỗi test (nhanh hơn 5-10 lần)

## Troubleshooting

Nếu gặp lỗi, kiểm tra:

1. Virtual environment đã được kích hoạt chưa?
2. GPU có được nhận diện không: `python -c "import torch; print(torch.cuda.is_available())"`
3. CUDA driver đã được cài đặt chưa: `nvidia-smi`

