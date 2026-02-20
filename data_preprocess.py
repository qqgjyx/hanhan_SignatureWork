#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_preprocess.py

将日度 FinBERT 情绪因子 (fed_sentiment_daily.csv) 合并到已有的
final_training_panel_with_rate.parquet 中，生成带有
fed_sent_pos / fed_sent_neg / fed_sent_neu / fed_sent_net 的新 panel。

运行方式:
    /usr/local/bin/python3 "/Users/wyhmac/Desktop/SW/data_preprocess.py"
"""

import os
from pathlib import Path
import pandas as pd


# ===================== 路径配置 =====================
BASE_DIR = Path("/Users/wyhmac/Desktop/SW")

PANEL_IN   = BASE_DIR / "final_training_panel_with_rate.parquet"
PANEL_BACK = BASE_DIR / "final_training_panel_with_rate_backup_before_sentiment.parquet"
PANEL_OUT  = BASE_DIR / "final_training_panel_with_rate.parquet"  # 覆盖写回

FED_SENT_DAILY = BASE_DIR / "fed_sentiment_daily.csv"


def main():
    # 1. 读取原始 panel
    print(f">>> loading base panel from: {PANEL_IN}")
    panel = pd.read_parquet(PANEL_IN)
    print(f"panel shape: {panel.shape}")

    # 2. 读取日度情绪
    print(f">>> loading daily sentiment from: {FED_SENT_DAILY}")
    sent = pd.read_csv(FED_SENT_DAILY)
    print(f"daily sentiment shape: {sent.shape}")

    # 3. 统一 date 格式（panel 中一般是 datetime64[ns] 或字符串）
    if "date" not in panel.columns:
        raise ValueError("Panel must contain a 'date' column.")

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.date

    sent = sent.copy()
    if "date" not in sent.columns:
        raise ValueError("Daily sentiment CSV must contain a 'date' column.")
    sent["date"] = pd.to_datetime(sent["date"], errors="coerce").dt.date

    # 4. 如果 panel 里已经有历史的 fed_sent_* 列，统一先删掉，避免冲突
    for col in ["fed_sent_pos", "fed_sent_neg", "fed_sent_neu", "fed_sent_net", "n_articles"]:
        if col in panel.columns:
            print(f"warning: column '{col}' already in panel, will be replaced from sentiment CSV.")
            panel = panel.drop(columns=[col])

    # 5. 左连接 merge（按日期），Fed 情绪是日度全市场/全 Fed 文本的共同信号
    print(">>> merging daily sentiment into panel by date ...")
    merged = panel.merge(sent, on="date", how="left")

    # 6. 保证模型需要的列一定存在，并填充缺失值（例如非 Fed 日 -> NaN）
    for col in ["fed_sent_pos", "fed_sent_neg", "fed_sent_neu", "fed_sent_net"]:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)

    if "n_articles" in merged.columns:
        merged["n_articles"] = merged["n_articles"].fillna(0).astype(int)

    print("merged panel shape:", merged.shape)

    # 7. 先备份原始 panel 再覆盖写回
    print(f">>> saving backup of original panel to: {PANEL_BACK}")
    panel.to_parquet(PANEL_BACK, index=False)

    print(f">>> writing merged panel (with FinBERT sentiment) to: {PANEL_OUT}")
    merged.to_parquet(PANEL_OUT, index=False)

    print("✅ done: panel updated with fed_sent_pos/neg/neu/net.")


if __name__ == "__main__":
    main()
