#!/usr/bin/env python3
"""Kiem tra metrics trong V7.1 artifact"""

import os
import torch

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
artifact_path = os.path.join(workspace_root, 'nvda_lstm_v7_multistock', 'nvda_lstm_v7_1_artifact.pth')

data = torch.load(artifact_path, map_location='cpu', weights_only=False)

metrics = data.get('test_metrics', {})
profit = data.get('test_profitability', {})

print("V7.1 Test Metrics (from artifact):")
print(f"  Buy Win Rate: {metrics.get('buy_wr', 0):.1%}")
print(f"  Sell Win Rate: {metrics.get('sell_wr', 0):.1%}")
print(f"  Combined Win Rate: {metrics.get('combined_wr', 0):.1%}")
print(f"  Coverage: {metrics.get('coverage', 0):.1%}")
print(f"  Buy Expectancy: {metrics.get('buy_expectancy', 0):.4f}")
print(f"  Sell Expectancy: {metrics.get('sell_expectancy', 0):.4f}")
print(f"\nProfitability:")
print(f"  Total Return: {profit.get('total_return', 0):.2%}")
print(f"  Profit Factor: {profit.get('profit_factor', 0):.2f}")
print(f"  Sharpe Ratio: {profit.get('sharpe_ratio', 0):.2f}")

