import sys
# 最先刷新一行，确认脚本已启动（若下面 import 或读 parquet 时发生段错误，至少能看到这一行）
print("panel_summary: starting ...", flush=True)
import numpy as np
import pandas as pd

# 立即刷新输出，便于在崩溃前看到进度（若读 parquet 时发生段错误则无后续输出）
def log(msg):
    print(msg, flush=True)

# ===== 1. 读取面板数据 =====
panel_path = "/Users/wyhmac/Desktop/SW/final_training_panel_with_rate.parquet"
log(f"Loading panel from: {panel_path}")
df = pd.read_parquet(panel_path)
log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

print("\n=== Basic shape & columns ===")
print("Shape (rows, cols):", df.shape)
print("First 10 columns:", df.columns.tolist()[:10])

# 确保 date 是 datetime
if not pd.api.types.is_datetime64_any_dtype(df["date"]):
    df["date"] = pd.to_datetime(df["date"])

# ===== 2. 时间维度信息 =====
print("\n=== Time coverage ===")
date_min = df["date"].min()
date_max = df["date"].max()
n_dates = df["date"].nunique()

print("Date range:", date_min.date(), "→", date_max.date())
print("Number of distinct trading days:", n_dates)

# ===== 3. 股票、行业、市值分布 =====
print("\n=== Universe summary ===")
n_tickers = df["ticker"].nunique()
n_sectors = df["sector"].nunique()
n_cap_buckets = df["cap_bucket"].nunique()

print("Number of distinct tickers:", n_tickers)
print("Number of sectors:", n_sectors)
print("Number of cap buckets:", n_cap_buckets)

print("\nTickers per sector (nunique):")
print(df.groupby("sector")["ticker"].nunique().sort_index())

print("\nTickers per cap_bucket (nunique):")
print(df.groupby("cap_bucket")["ticker"].nunique().sort_index())

# （可选） sector × cap_bucket 的交叉表
print("\nTicker counts by sector × cap_bucket (nunique):")
sector_cap_counts = (
    df.groupby(["sector", "cap_bucket"])["ticker"]
      .nunique()
      .unstack("cap_bucket")
      .fillna(0)
      .astype(int)
)
print(sector_cap_counts)

# ===== 4. 标签分布（整体 & 分行业 / 分市值） =====
# 面板里可能没有 "label"，而是有 "fwd_ret_60d"；与 train_ensemble_v5 一致：label = (fwd_ret_60d > 0)
if "label" not in df.columns and "fwd_ret_60d" in df.columns:
    df["label"] = np.where(df["fwd_ret_60d"].notna(), (df["fwd_ret_60d"] > 0).astype(int), np.nan)
label_col = "label" if "label" in df.columns else None

if label_col is None:
    print("\n=== Label: 无 'label' 或 'fwd_ret_60d' 列，跳过标签统计 ===")
    print("当前列名:", df.columns.tolist())
elif label_col not in df.columns:
    print("\n=== Label: 列 'label' 不存在，跳过标签统计 ===")
else:
    print(f"\n=== Label distribution (column = '{label_col}') ===")
    print(df[label_col].value_counts(dropna=False).sort_index())
    if df[label_col].notna().any():
        pos_rate = df[label_col].mean()
        print(f"Positive rate (label = 1) overall: {pos_rate:.4f}")

    print("\nPositive rate by sector:")
    print(
        df.groupby("sector")[label_col]
          .mean()
          .sort_index()
          .round(4)
    )

    print("\nPositive rate by cap_bucket:")
    print(
        df.groupby("cap_bucket")[label_col]
          .mean()
          .sort_index()
          .round(4)
    )

    # ===== 5. （可选）按时间粗略看一下标签均衡情况 =====
    print("\n=== Label balance by year-quarter (rough check) ===")
    df["year_quarter"] = df["date"].dt.to_period("Q")
    label_by_q = (
        df.groupby("year_quarter")[label_col]
          .mean()
          .to_frame("pos_rate")
          .round(4)
    )
    print(label_by_q)
