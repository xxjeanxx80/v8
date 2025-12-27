# So sanh ket qua V8 vs V7.1

## Tong quan

Script nay so sanh ket qua cua 2 versions:
- **V7.1**: No-leakage strategy (pretrain 2015-2020, fine-tune 2021-2024, test 2025)
- **V8**: Extended test period (pretrain 2015-2020, fine-tune 2021-2023, test 2024-2025)

## Strategy Differences

| Aspect | V7.1 | V8 |
|--------|------|----|
| Pretrain | 2015-2020 (5 nam) | 2015-2020 (5 nam) - GIONG |
| Fine-tune | 2021-2024 (4 nam) | 2021-2023 (3 nam) - NGAN HON |
| Validation | 2023-2024 | 2022-2023 - SOM HON |
| Test | 2025 (1 nam) | 2024-2025 (2 nam) - DAI HON |

## Metrics Comparison

| Metric | V8 | V7.1 | Difference |
|--------|----|----|------------|
| Buy Win Rate | 73.7% | 53.8% | +19.9% |
| Sell Win Rate | 50.0% | 0.0% | +50.0% |
| Combined Win Rate | 61.9% | 26.9% | +34.9% |
| Buy Expectancy | 0.0340 | -0.0115 | +0.0455 |
| Sell Expectancy | 0.0011 | 0.0000 | +0.0011 |
| Buy Coverage | 21.4% | 6.2% | +15.2% |
| Sell Coverage | 9.5% | 0.0% | +9.5% |
| Total Coverage | 31.0% | 6.2% | +24.8% |
| RMSE | 0.0683 | 0.0606 | +0.0078 |
| MAE | 0.0533 | 0.0467 | +0.0066 |

## V8 Metrics by Year (2024 vs 2025)

| Metric | 2024 | 2025 |
|--------|------|------|
| Buy Win Rate | 76.5% | 66.7% |
| Sell Win Rate | 66.7% | 38.5% |
| Combined Win Rate | 71.6% | 52.6% |
| Buy Coverage | 30.8% | 12.5% |
| Sell Coverage | 8.1% | 10.8% |

## Ket luan

Xem chi tiet trong file CSV: `v8_v7_1_comparison.csv`

### Key Insights:
- V8 co test period dai hon (2024-2025) so voi V7.1 (chi 2025)
- V8 fine-tune ngan hon (2021-2023) so voi V7.1 (2021-2024)
- V8 validation som hon (2022-2023) so voi V7.1 (2023-2024)
- V8 cho phep danh gia model tren 2 nam thay vi 1 nam
