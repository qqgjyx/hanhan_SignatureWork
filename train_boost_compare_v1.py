#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_boost_compare_v1.py

对比用的 Gradient Boosting (Boost) 模型：
- 读取 final_training_panel_with_rate.parquet
- 构造 rolling / EMA / 行业&市值 one-hot 特征（与 train_xgb_compare_v1.py / train_logistic_compare_v1.py 一致）
- 标签：相对 60 天收益（同一天截面上，高于中位数记为 1）
- 时间滚动切分：
    * 2025Q1: train <= 2024-12-31, val: 2025-01-01 ~ 2025-03-31
    * 2025Q2: train <= 2025-03-31, val: 2025-04-01 ~ 2025-06-30
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


# ==============================
# 1. 构造特征 & 标签 & 时间切分
#    和 XGB / Logistic 版本保持一致
# ==============================
def build_xy():
    BASE_DIR = Path("/Users/wyhmac/Desktop/SW")
    PANEL_PATH = BASE_DIR / "final_training_panel_with_rate.parquet"

    print(">>> loading panel from", PANEL_PATH)
    df = pd.read_parquet(PANEL_PATH)
    print("raw panel shape:", df.shape)

    # 规范日期 & 排序
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 只保留有 fwd_ret_60d 的样本
    if "fwd_ret_60d" not in df.columns:
        raise ValueError("fwd_ret_60d column not found in panel.")
    df = df[~df["fwd_ret_60d"].isna()].copy()

    # 日收益，如果没有的话算一下
    if "ret" not in df.columns:
        df["ret"] = df.groupby("ticker")["close"].pct_change()

    # === 每只股票做 rolling 特征 ===
    def add_roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()

        for W in [10, 20]:
            g[f"price_mean_{W}"] = g["close"].rolling(W, min_periods=5).mean()
            g[f"price_min_{W}"] = g["close"].rolling(W, min_periods=5).min()
            g[f"price_max_{W}"] = g["close"].rolling(W, min_periods=5).max()

            g[f"ret_mean_{W}"] = g["ret"].rolling(W, min_periods=5).mean()
            g[f"ret_std_{W}"] = g["ret"].rolling(W, min_periods=5).std()

            # 累计收益：(1+r1)*(1+r2)*...*(1+rW) - 1
            g[f"ret_cum_{W}"] = (
                (1.0 + g["ret"])
                .rolling(W, min_periods=5)
                .apply(lambda x: float(np.prod(x) - 1.0), raw=True)
            )

        # EMA 平滑
        g["close_ema_5"] = g["close"].ewm(span=5, adjust=False).mean()
        g["close_ema_10"] = g["close"].ewm(span=10, adjust=False).mean()
        g["close_ema_20"] = g["close"].ewm(span=20, adjust=False).mean()

        return g

    df = df.groupby("ticker", group_keys=False).apply(add_roll)

    # === 相对 60d 收益标签：同一天截面上，相对于中位数 ===
    df["_daily_median_60d"] = df.groupby("date")["fwd_ret_60d"].transform("median")
    df["y_rel"] = (df["fwd_ret_60d"] > df["_daily_median_60d"]).astype(int)
    df = df.drop(columns=["_daily_median_60d"])

    # rolling 特征列
    roll_cols = [
        "price_mean_10", "price_min_10", "price_max_10",
        "ret_mean_10", "ret_std_10", "ret_cum_10",
        "price_mean_20", "price_min_20", "price_max_20",
        "ret_mean_20", "ret_std_20", "ret_cum_20",
        "close_ema_5", "close_ema_10", "close_ema_20",
    ]
    roll_cols = [c for c in roll_cols if c in df.columns]

    # 把 rolling 前几行的 NaN 删掉
    if roll_cols:
        df = df.dropna(subset=roll_cols)

    # 行业 / 规模 -> category -> one-hot
    for col in ["sector", "cap_bucket"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df = pd.get_dummies(
        df,
        columns=[c for c in ["sector", "cap_bucket"] if c in df.columns],
        prefix=["sector", "cap"],
        drop_first=False,
    )

    # 数值特征（存在才用）
    base_num_cols = [
        "close", "volume", "ma15", "ret",
        "gold_price", "oil_price", "usd_index",
        "fed_sent_pos", "fed_sent_neg", "fed_sent_neu", "fed_sent_net",
        "fed_rate_hike", "fed_rate_cut", "fed_rate_hold",
        "fed_rate_decision_day", "fed_rate_change_bp",
        "corr_gold", "corr_oil", "corr_usd", "corr_fed",
    ]
    base_num_cols = [c for c in base_num_cols if c in df.columns]

    # one-hot 列
    cat_cols = [c for c in df.columns if c.startswith("sector_") or c.startswith("cap_")]

    feature_cols = base_num_cols + roll_cols + cat_cols

    # 去掉缺失
    df = df.dropna(subset=feature_cols + ["y_rel"])

    print("with features shape:", df.shape)
    print("num features:", len(feature_cols))

    # =========================
    # 时间切分（防信息泄露）
    # =========================
    cut_q1_train = pd.Timestamp("2024-12-31")
    cut_q1_val_end = pd.Timestamp("2025-03-31")
    cut_q2_train = cut_q1_val_end
    cut_q2_val_end = pd.Timestamp("2025-06-30")

    df_q1_train = df[df["date"] <= cut_q1_train]
    df_q1_val = df[(df["date"] > cut_q1_train) & (df["date"] <= cut_q1_val_end)]

    df_q2_train = df[df["date"] <= cut_q2_train]
    df_q2_val = df[(df["date"] > cut_q2_train) & (df["date"] <= cut_q2_val_end)]

    print("Q1 train size:", len(df_q1_train), "val size:", len(df_q1_val))
    print("Q2 train size:", len(df_q2_train), "val size:", len(df_q2_val))

    X_q1_train = df_q1_train[feature_cols].values
    y_q1_train = df_q1_train["y_rel"].values
    X_q1_val = df_q1_val[feature_cols].values
    y_q1_val = df_q1_val["y_rel"].values

    X_q2_train = df_q2_train[feature_cols].values
    y_q2_train = df_q2_train["y_rel"].values
    X_q2_val = df_q2_val[feature_cols].values
    y_q2_val = df_q2_val["y_rel"].values

    return (
        X_q1_train, y_q1_train, X_q1_val, y_q1_val,
        X_q2_train, y_q2_train, X_q2_val, y_q2_val,
        feature_cols,
    )


# ==============================
# 2. 跑 Gradient Boosting（Boost）
# ==============================
def run_stage(stage_name, X_train, y_train, X_val, y_val):
    print(f"\n===== BOOST {stage_name} =====")

    # 手工 class_weight -> sample_weight 来平衡类别
    n_total = len(y_train)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)

    # 防止除零
    if n_pos == 0 or n_neg == 0:
        print("Warning: only one class in y_train, skipping class weights.")
        sample_weight = None
    else:
        w_pos = n_total / (2.0 * n_pos)
        w_neg = n_total / (2.0 * n_neg)
        sample_weight = np.where(y_train == 1, w_pos, w_neg)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred)
    try:
        auc = roc_auc_score(y_val, prob)
    except ValueError:
        auc = float("nan")
    bal = balanced_accuracy_score(y_val, pred)

    print(
        f"BOOST {stage_name} @0.5 -> "
        f"acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}, bal_acc={bal:.4f}"
    )
    print("confusion_matrix:")
    print(confusion_matrix(y_val, pred))
    print("classification_report:")
    print(classification_report(y_val, pred, digits=4))


def main():
    (
        X_q1_train, y_q1_train, X_q1_val, y_q1_val,
        X_q2_train, y_q2_train, X_q2_val, y_q2_val,
        feature_cols,
    ) = build_xy()

    run_stage("2025Q1", X_q1_train, y_q1_train, X_q1_val, y_q1_val)
    run_stage("2025Q2", X_q2_train, y_q2_train, X_q2_val, y_q2_val)


if __name__ == "__main__":
    main()
