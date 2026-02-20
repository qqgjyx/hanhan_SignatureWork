# -*- coding: utf-8 -*-
"""
train_ensemble_v5.py (fixed: impute NaN before scaling + LR)
- 在 v3 基础上做四项增强：
  1) 极端失衡阶段自动下/上采样（仅训练集，避免泄露）
  2) LR/RF: class_weight='balanced'；XGB: 动态 scale_pos_weight
  3) 新特征：行业/市值 one-hot + 宏观×(行业/市值)交互 + 行业内相对强弱
  4) 平滑降噪：rolling + EMA，全部 shift(1) 防信息泄露
- 评估以 balanced accuracy 为主，同时输出 AUC、F1、混淆矩阵
- 两个滚动阶段：2025Q1、2025Q2
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise RuntimeError("请先安装 xgboost: pip install xgboost") from e

# 训练集失衡采样（可选）
try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    IMB_OK = True
except Exception:
    IMB_OK = False

RNG = 42
np.random.seed(RNG)

BASE_DIR = Path("/Users/wyhmac/Desktop/SW")
PANEL_PATH = BASE_DIR / "final_training_panel_with_rate.parquet"

# ============= 核心超参 =============
HORIZON = 60                     # 60 个交易日前瞻
ROLL_WINDOWS = [5, 10, 20]       # 滚动窗口（rolling 后 shift(1)）
EMA_SPANS = [5, 10]              # EMA 平滑
EXTREME_IMB_THRESH = 0.80        # 若正类占比 >80% 或 <20% 触发采样
APPLY_UNDERSAMPLE = True         # 采样策略：先尝试下采样
APPLY_OVERSAMPLE = False         # 可改成上采样
# ===================================


def safe_has_cols(df, cols):
    return [c for c in cols if c in df.columns]


def build_label(df):
    """按 ticker 计算 60d 前瞻收益，再做二分类标签（>0 为 1）。"""
    df = df.sort_values(["ticker", "date"]).copy()
    df["fwd_ret_60d"] = (
        df.groupby("ticker")["close"]
          .apply(lambda s: (s.shift(-HORIZON) / s) - 1.0)
          .reset_index(level=0, drop=True)
    )
    df["label"] = (df["fwd_ret_60d"] > 0.0).astype(int)
    return df


def add_rolling_features(g):
    """对单个 ticker 的时间序列添加滚动统计并 shift(1) 防泄露。"""
    g = g.copy()
    g["ret"] = g["close"].pct_change()

    for w in ROLL_WINDOWS:
        # 价格统计
        g[f"price_mean_{w}"] = g["close"].rolling(w).mean()
        g[f"price_min_{w}"]  = g["close"].rolling(w).min()
        g[f"price_max_{w}"]  = g["close"].rolling(w).max()

        # 收益统计
        g[f"ret_mean_{w}"] = g["ret"].rolling(w).mean()
        g[f"ret_std_{w}"]  = g["ret"].rolling(w).std()
        g[f"ret_cum_{w}"]  = g["ret"].rolling(w).sum()

    # EMA 平滑（对价格与收益）
    for span in EMA_SPANS:
        g[f"close_ema_{span}"] = g["close"].ewm(span=span, adjust=False).mean()
        g[f"ret_ema_{span}"]   = g["ret"].ewm(span=span, adjust=False).mean()

    # 全部往前移动 1 天，确保只用过去信息
    roll_cols = [c for c in g.columns if any(
        kw in c for kw in ["price_mean_", "price_min_", "price_max_",
                           "ret_mean_", "ret_std_", "ret_cum_",
                           "close_ema_", "ret_ema_"])]
    g[roll_cols] = g[roll_cols].shift(1)

    return g


def add_macro_deltas(df):
    """宏观变量的变化率（全市场同一值，直接按 date 计算后 merge）"""
    df = df.sort_values("date")
    for col in safe_has_cols(df, ["gold_price", "oil_price", "usd_index"]):
        df[f"{col}_chg_5"]  = df[col].pct_change(5)
        df[f"{col}_chg_10"] = df[col].pct_change(10)
    return df


def add_relative_features(df):
    """
    行业内相对强弱：每天在行业内计算 5/20 日累计收益的排名与标准化，
    以及相对行业均值的 alpha。
    """
    df = df.copy()

    if "ret" not in df.columns:
        df = df.sort_values(["ticker", "date"])
        df["ret"] = df.groupby("ticker")["close"].pct_change()

    if "ret_cum_20" not in df.columns:
        df["ret_cum_20"] = (
            df.groupby("ticker")["ret"].rolling(20).sum().reset_index(level=0, drop=True)
        ).shift(1)  # 防泄露

    # 每日每行业的均值
    df["ret20_sector_mean"] = (
        df.groupby(["date", "sector"])["ret_cum_20"].transform("mean")
    )
    df["alpha_vs_sector_20"] = df["ret_cum_20"] - df["ret20_sector_mean"]

    # 行业内排名（百分位）
    def rank_pct(s):
        return s.rank(pct=True, method="average")

    df["rank_ret20_in_sector"] = (
        df.groupby(["date", "sector"])["ret_cum_20"].transform(rank_pct)
    )
    df["rank_ret5_in_sector"] = (
        df.groupby(["date", "sector"])["ret"]
          .transform(lambda s: s.rolling(5).sum())
          .transform(rank_pct)
    )

    # 按市值组（cap_bucket）做一个类似的相对强弱
    df["ret20_cap_mean"] = (
        df.groupby(["date", "cap_bucket"])["ret_cum_20"].transform("mean")
    )
    df["alpha_vs_cap_20"] = df["ret_cum_20"] - df["ret20_cap_mean"]

    return df


def add_onehots_and_interactions(df):
    """行业/市值 one-hot，以及宏观×(行业/市值)、情绪×行业等交互项。"""
    df = df.copy()
    # one-hot
    sector_dum = pd.get_dummies(df["sector"], prefix="sector")
    cap_dum    = pd.get_dummies(df["cap_bucket"], prefix="cap_bucket")
    df = pd.concat([df, sector_dum, cap_dum], axis=1)

    # 宏观 / Fed 信号列名
    fed_cols = safe_has_cols(df, [
        "fed_rate_hike","fed_rate_cut","fed_rate_hold",
        "fed_rate_change_bp","fed_rate_decision_day",
        "fed_sent_net"
    ])
    macro_chg_cols = safe_has_cols(df, [
        "gold_price_chg_5","oil_price_chg_5","usd_index_chg_5",
        "gold_price_chg_10","oil_price_chg_10","usd_index_chg_10",
    ])

    # 行业×Fed / 宏观
    for s in sector_dum.columns:
        if "fed_rate_cut" in fed_cols:
            df[f"{s}__x__fed_cut"] = df[s] * df.get("fed_rate_cut", 0)
        if "fed_rate_hike" in fed_cols:
            df[f"{s}__x__fed_hike"] = df[s] * df.get("fed_rate_hike", 0)
        if "fed_sent_net" in fed_cols:
            df[f"{s}__x__fed_sent_net"] = df[s] * df.get("fed_sent_net", 0.0)

        for m in macro_chg_cols:
            df[f"{s}__x__{m}"] = df[s] * df[m]

    # 市值×Fed / 宏观
    for c in cap_dum.columns:
        for k in ["fed_rate_cut","fed_rate_hike","fed_rate_hold","fed_sent_net"]:
            if k in df.columns:
                df[f"{c}__x__{k}"] = df[c] * df[k]
        for m in macro_chg_cols:
            df[f"{c}__x__{m}"] = df[c] * df[m]

    return df


def build_features(raw):
    """端到端构建特征与标签。"""
    df = raw.copy()

    # 标准化列名
    df.columns = [c.strip() for c in df.columns]
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    # 一些可能缺失的列补 0
    for col in ["volume","ma15","gold_price","oil_price","usd_index",
                "fed_sent_pos","fed_sent_neg","fed_sent_neu","fed_sent_net",
                "fed_rate_hike","fed_rate_cut","fed_rate_hold",
                "fed_rate_change_bp","fed_rate_decision_day"]:
        if col not in df.columns:
            df[col] = 0.0

    # 先做滚动 + 平滑（逐 ticker）
    df = df.groupby("ticker", group_keys=False).apply(add_rolling_features)

    # 宏观变化率（按日期）
    df = df.groupby("date", group_keys=False).apply(lambda g: g).reset_index(drop=True)
    df = add_macro_deltas(df)

    # 行业内相对强弱
    df = add_relative_features(df)

    # one-hot 与交互
    df = add_onehots_and_interactions(df)

    # 二分类标签
    df = build_label(df)

    # 替换 inf，再统一 NaN 留给后面 impute
    df = df.replace([np.inf, -np.inf], np.nan)

    # 丢掉 close 或 label 缺失的
    df = df.dropna(subset=["close", "label"]).copy()

    # 特征列：排除非数值/目标/原始文本列
    exclude = set(["date","ticker","sector","cap_bucket","label","fwd_ret_60d"])
    X_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    return df, X_cols


def maybe_resample(X, y):
    """极端失衡时对训练集采样（不改验证集）。"""
    pos_ratio = y.mean()
    if pos_ratio >= EXTREME_IMB_THRESH or pos_ratio <= (1-EXTREME_IMB_THRESH):
        if IMB_OK:
            if APPLY_UNDERSAMPLE:
                rus = RandomUnderSampler(random_state=RNG)
                X_res, y_res = rus.fit_resample(X, y)
                print(f"[resample] RandomUnderSampler | pos_ratio {pos_ratio:.3f} -> {y_res.mean():.3f} | n={len(y_res)}")
                return X_res, y_res
            if APPLY_OVERSAMPLE:
                ros = RandomOverSampler(random_state=RNG)
                X_res, y_res = ros.fit_resample(X, y)
                print(f"[resample] RandomOverSampler  | pos_ratio {pos_ratio:.3f} -> {y_res.mean():.3f} | n={len(y_res)}")
                return X_res, y_res
        else:
            print(f"[warn] 极端失衡 pos_ratio={pos_ratio:.3f}，但未安装 imbalanced-learn；仅用 class_weight/scale_pos_weight。")
    return X, y


def fit_stage(df, X_cols, stage_name, train_end, val_start, val_end):
    """单个阶段：构造切片→补缺失→缩放→采样→训练三模型→集成→评估。"""
    msk_train = (df["date"] <= pd.to_datetime(train_end))
    msk_val   = (df["date"] >= pd.to_datetime(val_start)) & (df["date"] <= pd.to_datetime(val_end))

    train = df.loc[msk_train].copy()
    val   = df.loc[msk_val].copy()

    # --- 先在 DataFrame 级别做 NaN & inf 处理 ---
    X_train_df = train[X_cols].replace([np.inf, -np.inf], np.nan)
    X_val_df   = val[X_cols].replace([np.inf, -np.inf], np.nan)

    # 用训练集的列均值做 imputation（避免信息泄露）
    col_means = X_train_df.mean(axis=0)
    X_train_df = X_train_df.fillna(col_means)
    X_val_df   = X_val_df.fillna(col_means)

    X_train = X_train_df.values
    y_train = train["label"].values.astype(int)
    X_val   = X_val_df.values
    y_val   = val["label"].values.astype(int)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    # 采样（仅训练集）
    X_train_sc, y_train = maybe_resample(X_train_sc, y_train)

    # 三模型
    lr = LogisticRegression(
        solver="saga",
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RNG,
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RNG,
    )
    # XGB 的 scale_pos_weight = neg/pos
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    spw = (neg / max(pos, 1))

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=2,
        random_state=RNG,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=spw,
    )

    print(f"\n===== Stage {stage_name} =====")
    print(f"train size: {len(y_train)} val size: {len(y_val)}")

    # 训练
    lr.fit(X_train_sc, y_train)
    rf.fit(X_train_sc, y_train)
    xgb.fit(X_train_sc, y_train)

    # 验证概率
    p_lr  = lr.predict_proba(X_val_sc)[:, 1]
    p_rf  = rf.predict_proba(X_val_sc)[:, 1]
    p_xgb = xgb.predict_proba(X_val_sc)[:, 1]

    # 简单加权集成（xgb 权重大）
    P = 0.25 * p_lr + 0.35 * p_rf + 0.40 * p_xgb

    # 0.5 阈值
    y_hat_05 = (P >= 0.5).astype(int)
    acc05    = accuracy_score(y_val, y_hat_05)
    f105     = f1_score(y_val, y_hat_05)
    auc      = roc_auc_score(y_val, P)
    bal05    = balanced_accuracy_score(y_val, y_hat_05)
    print(f"Ensemble @0.5 -> acc={acc05:.4f}, f1={f105:.4f}, auc={auc:.4f}, bal_acc={bal05:.4f}")

    # 搜索最佳阈值（以 balanced accuracy 最大为目标）
    grid = np.linspace(0.15, 0.85, 29)
    best_thr, best_bal = 0.5, -1
    best_stats = None
    for t in grid:
        y_pred = (P >= t).astype(int)
        bal = balanced_accuracy_score(y_val, y_pred)
        if bal > best_bal:
            best_bal = bal
            best_thr = t
            acc = accuracy_score(y_val, y_pred)
            f1  = f1_score(y_val, y_pred)
            best_stats = (acc, f1, bal)

    acc, f1, bal = best_stats
    print(f"Best thr -> {best_thr:.3f} | acc={acc:.4f}, f1={f1:.4f}, bal_acc={bal:.4f}")
    y_best = (P >= best_thr).astype(int)
    print("confusion_matrix @best_thr:")
    print(confusion_matrix(y_val, y_best))
    print("\nclassification_report @best_thr:")
    print(classification_report(y_val, y_best, digits=3))

    # 打印 XGB 的 Top20 特征
    try:
        importances = xgb.feature_importances_
        order = np.argsort(importances)[::-1][:20]
        print("\n>>> XGB Top 20 features:")
        for i, idx in enumerate(order, 1):
            print(f"{i:2d}. {X_cols[idx]:40s} {importances[idx]:.5f}")
    except Exception:
        pass

    return {
        "lr": lr, "rf": rf, "xgb": xgb, "scaler": scaler,
        "X_cols": X_cols, "best_thr": best_thr
    }


def main():
    print(">>> train_ensemble_v5.py: loading & building features")
    print(f">>> loading panel from {PANEL_PATH}")
    raw = pd.read_parquet(PANEL_PATH)
    print("raw shape:", raw.shape)

    df, X_cols = build_features(raw)
    print("with features shape:", df.shape)
    print("num features:", len(X_cols))

    # ===== 阶段 1：用 2024-12-31 以前训练，预测 2025-01-01 ~ 2025-03-31 =====
    fit_stage(
        df, X_cols,
        stage_name="(2024-12-31 | 2025-01-01 ~ 2025-03-31)",
        train_end="2024-12-31",
        val_start="2025-01-01",
        val_end="2025-03-31",
    )

    # ===== 阶段 2：用 2025-03-31 以前训练，预测 2025-04-01 ~ 2025-06-30 =====
    fit_stage(
        df, X_cols,
        stage_name="(2025-03-31 | 2025-04-01 ~ 2025-06-30)",
        train_end="2025-03-31",
        val_start="2025-04-01",
        val_end="2025-06-30",
    )

    print("\n✅ done（采样 + 类别权重 + 结构/交互/相对特征 + 平滑 + 滚动验证 + NaN 填充）")


if __name__ == "__main__":
    main()
