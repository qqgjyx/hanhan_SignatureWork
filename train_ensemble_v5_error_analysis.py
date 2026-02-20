#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_ensemble_v5_error_analysis.py

基于你现有的 final_training_panel_with_rate.parquet，做三件事：

1. 构建丰富特征（价格/收益 rolling、宏观、Fed 情绪 & 利率、行业 & 市值 one-hot、交互项等）
2. 在 2025Q1 / 2025Q2 做时间滚动验证，训练一个简单的三模型集成：
   - LogisticRegression（class_weight="balanced"）
   - RandomForestClassifier（class_weight="balanced"）
   - XGBClassifier（scale_pos_weight）
   集成概率 = 三个模型平均
3. 对每个阶段做“错误分析”：
   - 各行业错误率、FP/FN 占比
   - 各市值桶错误率
   - 每日错误率 + 5 日滚动
   - 特征维度上：错误样本 vs 正确样本的均值差异（Top 20）
   - 将完整验证集（含预测、错误标签、所有特征）保存为 CSV

label 定义为：fwd_ret_60d > 0  →  1（未来 60 日收益为正），否则 0。
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# =========================
# 路径设置
# =========================

BASE_DIR = Path("/Users/wyhmac/Desktop/SW")
PANEL_PATH = BASE_DIR / "final_training_panel_with_rate.parquet"
OUTPUT_DIR = BASE_DIR / "error_analysis_v5"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 特征构造
# =========================

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    对单个 ticker 做按时间排序的 rolling 特征：
    - price_max/min/mean_{5,10,20}
    - ret_std_{5,10,20}
    - ret_cum_{10,20}
    - close 的 EMA（5,10,20）
    """
    df = df.sort_values("date").copy()

    # 确保 ret 存在；如果已经有就直接用
    if "ret" not in df.columns:
        df["ret"] = df["close"].pct_change()

    windows_price = [5, 10, 20]
    for w in windows_price:
        df[f"price_max_{w}"] = df["close"].rolling(w).max()
        df[f"price_min_{w}"] = df["close"].rolling(w).min()
        df[f"price_mean_{w}"] = df["close"].rolling(w).mean()

    windows_ret = [5, 10, 20]
    for w in windows_ret:
        df[f"ret_std_{w}"] = df["ret"].rolling(w).std()

    df["ret_cum_10"] = df["ret"].rolling(10).sum()
    df["ret_cum_20"] = df["ret"].rolling(20).sum()

    # 简单 EMA（用 pandas ewm）
    df["close_ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["close_ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["close_ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    return df


def build_sector_cap_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    通过 sector / cap_bucket 的相对收益构造一些“相对特征”：
    - ret20_sector_mean
    - ret20_cap_mean
    - alpha_vs_cap_20（个股 20 日累计收益 - 同市值组平均 20 日累计收益）
    """

    df = df.copy()

    # 先确保 ret 存在
    if "ret" not in df.columns:
        df["ret"] = df["close"].pct_change()

    # 20 日累计收益
    df["ret_cum_20"] = df.groupby("ticker")["ret"].rolling(20).sum().reset_index(level=0, drop=True)

    # 每日按 sector / cap_bucket 的平均 20 日累计收益
    df["ret20_sector_mean"] = (
        df.groupby(["date", "sector"])["ret_cum_20"].transform("mean")
    )
    df["ret20_cap_mean"] = (
        df.groupby(["date", "cap_bucket"])["ret_cum_20"].transform("mean")
    )

    # 相对市值组 alpha
    df["alpha_vs_cap_20"] = df["ret_cum_20"] - df["ret20_cap_mean"]

    return df


def build_full_features(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    从原始 panel 构造用于训练的特征 DataFrame：

    保留的基础字段：
      - date, ticker, sector, cap_bucket
      - close, volume, ma15
      - 宏观：gold_price, oil_price, usd_index
      - Fed 情绪：fed_sent_pos/neg/neu/net（如存在）
      - Fed 利率事件：fed_rate_*（如存在）
      - 相关性：corr_*（如存在）
      - ret, fwd_ret_60d

    然后构造：
      - rolling price/ret 特征
      - sector / cap_bucket 的相对特征
      - sector / cap_bucket one-hot
      - fed_sent_net × sector / cap_bucket 的交互项
    """

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 只保留我们需要的列（尽量兼容，缺了就忽略）
    base_cols = [
        "date",
        "ticker",
        "sector",
        "cap_bucket",
        "close",
        "volume",
        "ma15",
        "gold_price",
        "oil_price",
        "usd_index",
        "fed_sent_pos",
        "fed_sent_neg",
        "fed_sent_neu",
        "fed_sent_net",
        "corr_fed",
        "corr_usd",
        "corr_oil",
        "corr_gold",
        "corr_sector",
        "corr_cap",
        "fed_rate_hike",
        "fed_rate_cut",
        "fed_rate_hold",
        "fed_rate_change_bp",
        "fed_rate_decision_day",
        "ret",
        "fwd_ret_60d",
    ]

    existing_cols = [c for c in base_cols if c in df.columns]
    df = df[existing_cols].copy()

    # 确保 ret 存在
    if "ret" not in df.columns:
        df["ret"] = (
            df.sort_values(["ticker", "date"])
              .groupby("ticker")["close"]
              .pct_change()
        )

    # label：未来 60 日收益 > 0
    df["label"] = (df["fwd_ret_60d"] > 0).astype(int)

    # =========================
    # rolling 特征（按 ticker）
    # =========================
    print(">>> adding rolling features per ticker ...")
    df = df.groupby("ticker", group_keys=False).apply(add_rolling_features)

    # =========================
    # sector / cap 相对特征
    # =========================
    print(">>> adding sector / cap relative features ...")
    df = build_sector_cap_relative_features(df)

    # =========================
    # one-hot: sector / cap_bucket
    # =========================
    print(">>> adding sector / cap one-hot ...")
    sector_dum = pd.get_dummies(df["sector"], prefix="sector")
    cap_dum = pd.get_dummies(df["cap_bucket"], prefix="cap_bucket")

    df = pd.concat([df, sector_dum, cap_dum], axis=1)

    # 交互：fed_sent_net × sector / cap_bucket
    if "fed_sent_net" in df.columns:
        for col in sector_dum.columns:
            df[f"{col}__x__fed_sent_net"] = df[col] * df["fed_sent_net"]
        for col in cap_dum.columns:
            df[f"{col}__x__fed_sent_net"] = df[col] * df["fed_sent_net"]

    # =========================
    # 选择特征列
    # =========================
    # 非特征列：时间 / id / label / fwd_ret_60d
    non_feature_cols = {
        "date",
        "ticker",
        "sector",
        "cap_bucket",
        "fwd_ret_60d",
        "label",
    }

    feature_cols = [
        c for c in df.columns
        if (c not in non_feature_cols) and (df[c].dtype != "O")
    ]

    # 删掉那些全是 NaN 的特征
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    # 去掉有 NaN 的行（对所有特征 & label）
    full_cols = feature_cols + ["date", "ticker", "sector", "cap_bucket", "label"]
    df = df[full_cols].dropna().reset_index(drop=True)

    print(f"with features shape: {df.shape}")
    print(f"num features: {len(feature_cols)}")

    return df, feature_cols


# =========================
# 错误分析
# =========================

def analyze_errors(
    stage_name: str,
    df_val_raw: pd.DataFrame,
    X_val: np.ndarray,
    y_val: np.ndarray,
    p_val: np.ndarray,
    feature_cols: List[str],
    thr: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对某个阶段的验证集做错误分析，并返回：
      - df_val_full: 带预测结果 & 错误标签 & 全部特征
      - stats_df: 按特征计算的“错误 vs 正确”均值差异
    """

    print(f"\n===== Error analysis for {stage_name} (thr={thr:.3f}) =====")

    df_val = df_val_raw.copy().reset_index(drop=True)
    df_feat = pd.DataFrame(X_val, columns=feature_cols)

    df_val["y_true"] = y_val
    df_val["p_ens"] = p_val
    df_val["y_pred"] = (df_val["p_ens"] >= thr).astype(int)
    df_val["error"] = (df_val["y_pred"] != df_val["y_true"]).astype(int)

    df_val["err_type"] = "OK"
    df_val.loc[(df_val["y_true"] == 0) & (df_val["y_pred"] == 1), "err_type"] = "FP"
    df_val.loc[(df_val["y_true"] == 1) & (df_val["y_pred"] == 0), "err_type"] = "FN"

    # 拼上特征
    df_val_full = pd.concat([df_val, df_feat], axis=1)

    # 1) 按行业错误率
    if "sector" in df_val_full.columns:
        print("\n-- Error rate by sector --")
        print(df_val_full.groupby("sector")["error"].mean().sort_values(ascending=False))

        print("\n-- FP / FN share by sector --")
        pivot = (
            df_val_full
            .pivot_table(index="sector",
                         columns="err_type",
                         values="y_true",
                         aggfunc="count",
                         fill_value=0)
        )
        pivot["total"] = pivot.sum(axis=1)
        for col in ["FP", "FN", "OK"]:
            if col in pivot.columns:
                pivot[col + "_rate"] = pivot[col] / pivot["total"]
        print(pivot.sort_values("FP_rate", ascending=False))

    # 2) 按市值错误率
    if "cap_bucket" in df_val_full.columns:
        print("\n-- Error rate by cap_bucket --")
        print(df_val_full.groupby("cap_bucket")["error"].mean().sort_values(ascending=False))

    # 3) 每日错误率
    if "date" in df_val_full.columns:
        print("\n-- Daily error rate (head) --")
        daily_err = (
            df_val_full
            .groupby("date")["error"]
            .mean()
            .rename("err_rate")
        )
        print(daily_err.head())

        print("\nDaily error rolling(5) mean (head):")
        print(daily_err.rolling(5).mean().dropna().head())

    # 4) 连续特征：错误 vs 正确的均值差异
    num_cols = []
    for c in feature_cols:
        if np.issubdtype(df_val_full[c].dtype, np.number):
            num_cols.append(c)

    stats = []
    mask_err = df_val_full["error"] == 1
    mask_ok = df_val_full["error"] == 0

    for c in num_cols:
        m_err = df_val_full.loc[mask_err, c].mean()
        m_ok = df_val_full.loc[mask_ok, c].mean()
        delta = m_err - m_ok
        stats.append(
            {
                "feature": c,
                "mean_error": m_err,
                "mean_correct": m_ok,
                "delta": delta,
                "abs_delta": abs(delta),
            }
        )

    stats_df = pd.DataFrame(stats)

    if stats_df.empty:
        print("\n(No numeric feature stats computed; skip feature-diff table.)")
    else:
        if "abs_delta" not in stats_df.columns:
            stats_df["abs_delta"] = stats_df["delta"].abs()
        stats_df = stats_df.sort_values("abs_delta", ascending=False)

        print("\n-- Top 20 features with biggest error vs correct mean difference --")
        print(stats_df.head(20))

        # 保存特征差异表
        out_stats_csv = OUTPUT_DIR / f"error_analysis_{stage_name}_feature_diff.csv"
        stats_df.to_csv(out_stats_csv, index=False)
        print(f"✅ saved feature mean diffs to: {out_stats_csv}")

    # 保存完整验证集
    out_csv = OUTPUT_DIR / f"error_analysis_{stage_name}_full.csv"
    df_val_full.to_csv(out_csv, index=False)
    print(f"\n✅ saved full validation with errors to: {out_csv}")

    return df_val_full, stats_df


# =========================
# 训练 & 评估 & 错误分析
# =========================

def fit_and_eval_stage(
    stage_name: str,
    df_stage: pd.DataFrame,
    feature_cols: List[str],
    train_end: str,
    val_start: str,
    val_end: str,
) -> None:
    """
    对某个时间阶段做：
      - 切分 train / val（时间滚动，无信息泄露）
      - 训练 LR + RF + XGB
      - 计算 ensemble 指标（0.5 阈值 + 搜索最佳阈值）
      - 做错误分析并保存结果
    """

    df = df_stage.copy()
    df["date"] = pd.to_datetime(df["date"])

    train_end_ts = pd.Timestamp(train_end)
    val_start_ts = pd.Timestamp(val_start)
    val_end_ts = pd.Timestamp(val_end)

    train_mask = df["date"] <= train_end_ts
    val_mask = (df["date"] >= val_start_ts) & (df["date"] <= val_end_ts)

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_val = df.loc[val_mask].reset_index(drop=True)

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values

    X_val = df_val[feature_cols].values
    y_val = df_val["label"].values

    print(f"\n===== Stage {stage_name} ({train_end} | {val_start} ~ {val_end}) =====")
    print(f"train size: {len(df_train)} val size: {len(df_val)}")

    # 标准化
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # 类别比例，用于 XGB 的 scale_pos_weight
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    if pos == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = neg / pos

    # ============ 模型定义 ============
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )

    # ============ 拟合 ============
    lr.fit(X_train_sc, y_train)
    rf.fit(X_train_sc, y_train)
    xgb.fit(X_train_sc, y_train)

    # ============ 预测 ============
    p_lr = lr.predict_proba(X_val_sc)[:, 1]
    p_rf = rf.predict_proba(X_val_sc)[:, 1]
    p_xgb = xgb.predict_proba(X_val_sc)[:, 1]

    p_ens = (p_lr + p_rf + p_xgb) / 3.0

    # 在 0.5 阈值下先看一眼
    y_pred_05 = (p_ens >= 0.5).astype(int)
    acc_05 = accuracy_score(y_val, y_pred_05)
    f1_05 = f1_score(y_val, y_pred_05)
    auc = roc_auc_score(y_val, p_ens)
    bal_acc_05 = balanced_accuracy_score(y_val, y_pred_05)

    print(
        f"{stage_name} Ensemble @0.5 -> "
        f"acc={acc_05:.4f}, f1={f1_05:.4f}, auc={auc:.4f}, bal_acc={bal_acc_05:.4f}"
    )

    # ============ 搜索一个“最佳阈值”（按 balanced accuracy 最大） ============
    best_thr = 0.5
    best_bal_acc = -1.0
    best_f1 = 0.0

    for thr in np.linspace(0.2, 0.8, 61):  # step 0.01
        y_pred = (p_ens >= thr).astype(int)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        f1_val = f1_score(y_val, y_pred)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thr = thr
            best_f1 = f1_val

    y_pred_best = (p_ens >= best_thr).astype(int)
    acc_best = accuracy_score(y_val, y_pred_best)
    cm = confusion_matrix(y_val, y_pred_best)

    print(
        f"Best thr -> {best_thr:.3f} | "
        f"acc={acc_best:.4f}, f1={best_f1:.4f}, bal_acc={best_bal_acc:.4f}"
    )
    print("confusion_matrix @best_thr:")
    print(cm)
    print("\nclassification_report @best_thr:")
    print(classification_report(y_val, y_pred_best, digits=4))

    # ============ 错误分析 ============
    analyze_errors(
        stage_name=stage_name,
        df_val_raw=df_val[["date", "ticker", "sector", "cap_bucket"]].copy(),
        X_val=X_val,
        y_val=y_val,
        p_val=p_ens,
        feature_cols=feature_cols,
        thr=best_thr,
    )


# =========================
# main
# =========================

def main():
    print(">>> train_ensemble_v5_error_analysis.py: loading & building features")
    print(f">>> loading panel from {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    print(f"raw panel shape: {panel.shape}")

    df_feat, feature_cols = build_full_features(panel)

    # 2025Q1: train <= 2024-12-31, val 2025-01-01 ~ 2025-03-31
    fit_and_eval_stage(
        stage_name="2025Q1",
        df_stage=df_feat,
        feature_cols=feature_cols,
        train_end="2024-12-31",
        val_start="2025-01-01",
        val_end="2025-03-31",
    )

    # 2025Q2: train <= 2025-03-31, val 2025-04-01 ~ 2025-06-30
    fit_and_eval_stage(
        stage_name="2025Q2",
        df_stage=df_feat,
        feature_cols=feature_cols,
        train_end="2025-03-31",
        val_start="2025-04-01",
        val_end="2025-06-30",
    )

    print("\n✅ done (ensemble + error analysis for 2025Q1 & 2025Q2)")


if __name__ == "__main__":
    main()
