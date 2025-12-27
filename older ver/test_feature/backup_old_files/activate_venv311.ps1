# Script de kich hoat virtual environment Python 3.11 voi GPU support
# Chay script nay truoc khi chay cac test

Write-Host "Dang kich hoat virtual environment Python 3.11..." -ForegroundColor Green
& ".venv311\Scripts\Activate.ps1"

Write-Host "`nKiem tra GPU..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`nVirtual environment da san sang!" -ForegroundColor Green
Write-Host "Ban co the chay cac test ngay bay gio:" -ForegroundColor Cyan
Write-Host "  python comprehensive_test_runner.py" -ForegroundColor White

