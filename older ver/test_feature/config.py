#!/usr/bin/env python3
"""
File cấu hình chung cho tất cả các test feature
Quản lý tất cả các parameters, thresholds, metrics và paths
"""

import os

# ==================== Paths ====================
DATA_DIR = "../data"
RESULTS_DIR = "results"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tạo thư mục results nếu chưa tồn tại
if not os.path.exists(os.path.join(BASE_DIR, RESULTS_DIR)):
    os.makedirs(os.path.join(BASE_DIR, RESULTS_DIR))

# ==================== Model Parameters ====================
SEQUENCE_LENGTH = 30
HORIZON = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.0005
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 32

# ==================== Device Configuration ====================
# Tu dong detect va su dung GPU neu co
# Note: Import torch o day de tranh circular import
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU = torch.cuda.is_available()
except (ImportError, AttributeError):
    # Fallback neu torch chua duoc cai dat hoac khong co CUDA
    try:
        import torch
        DEVICE = torch.device("cpu")
        USE_GPU = False
    except ImportError:
        DEVICE = None
        USE_GPU = False

# ==================== Threshold Optimization ====================
# Các percentile thresholds để test
THRESHOLD_PERCENTILES = [50, 60, 70, 75, 80, 85, 90, 95]

# Threshold mặc định hiện tại (75th percentile)
DEFAULT_THRESHOLD_PERCENTILE = 75

# ==================== Feature Groups ====================
FEATURE_GROUPS = {
    'trend': ['price_vs_sma50', 'price_vs_sma200'],
    'momentum': ['rsi14', 'macd', 'macd_bullish', 'macd_signal', 'macd_hist'],
    'volatility': ['atr', 'bb_bandwidth', 'bb_percent', 'bb_upper', 'bb_lower', 'bb_middle'],
    'volume': ['volume_ratio', 'obv', 'volume_sma20'],
    'returns': ['daily_return', 'price_change', 'return_3d', 'return_5d', 'return_10d', 'return_20d'],
    'structure': ['hl_spread', 'hl_spread_pct', 'oc_spread', 'oc_spread_pct'],
    'position': ['bb_squeeze', 'rsi_overbought', 'rsi_oversold'],
    'sector': ['sox_beta', 'sox_correlation', 'beta_to_sox', 'rolling_corr_sox', 'stock_vs_sox', 'sox_return', 'sector_momentum']
}

# Feature set tốt nhất từ ablation test (no_trend)
BEST_FEATURE_SET_BASE = 'no_trend'

# ==================== Metrics Configuration ====================
# Các metrics chính để đánh giá
PRIMARY_METRICS = [
    'buy_win_rate',
    'sell_win_rate',
    'combined_win_rate',
    'coverage',
    'rmse',
    'mae'
]

# Các metrics trading bổ sung
TRADING_METRICS = [
    'total_return',
    'sharpe_ratio',
    'max_drawdown',
    'win_rate_by_strength',
    'false_positive_rate',
    'false_negative_rate'
]

# ==================== Tiêu chí lựa chọn Best Configuration ====================
MIN_COMBINED_WIN_RATE = 0.70
MIN_BUY_WIN_RATE = 0.75
MIN_SELL_WIN_RATE = 0.60
MAX_RMSE_INCREASE_PCT = 10  # RMSE không tăng quá 10% so với baseline
MIN_COVERAGE = 0.20  # Coverage tối thiểu 20%
MAX_COVERAGE = 1.0   # Coverage tối đa 100%

# ==================== Cross-Validation ====================
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# ==================== Feature Combination Testing ====================
# Forward selection: Bắt đầu từ core features
CORE_FEATURES = [
    'rsi14',
    'macd',
    'atr',
    'bb_bandwidth',
    'bb_percent',
    'volume_ratio',
    'obv',
    'hl_spread_pct',
    'daily_return',
    'price_change',
    'bb_squeeze'
]

# ==================== Trading Strategy Evaluation ====================
# Risk-free rate để tính Sharpe ratio (annual)
RISK_FREE_RATE = 0.02  # 2% per year

# Số ngày trading trong năm
TRADING_DAYS_PER_YEAR = 252

# ==================== Visualization ====================
FIGURE_SIZE = (12, 8)
DPI = 300
HEATMAP_CMAP = 'RdBu_r'

# ==================== Output Files ====================
OUTPUT_FILES = {
    'threshold_optimization': os.path.join(RESULTS_DIR, 'threshold_optimization_results.csv'),
    'threshold_heatmap': os.path.join(RESULTS_DIR, 'threshold_optimization_heatmap.png'),
    'feature_combination': os.path.join(RESULTS_DIR, 'feature_combination_results.csv'),
    'feature_importance': os.path.join(RESULTS_DIR, 'feature_importance_ranking.csv'),
    'advanced_ablation': os.path.join(RESULTS_DIR, 'advanced_ablation_results.csv'),
    'feature_contribution': os.path.join(RESULTS_DIR, 'feature_contribution_analysis.csv'),
    'trading_strategy': os.path.join(RESULTS_DIR, 'trading_strategy_report.csv'),
    'trading_performance': os.path.join(RESULTS_DIR, 'trading_performance_chart.png'),
    'comprehensive_report': os.path.join(RESULTS_DIR, 'comprehensive_test_report.md'),
    'best_configuration': os.path.join(RESULTS_DIR, 'best_configuration.json'),
    'recommendations': os.path.join(RESULTS_DIR, 'recommendations.txt')
}

# ==================== Logging ====================
VERBOSE = True
PRINT_EVERY_N_EPOCHS = 20

