#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm_compare_v1.py

对比用的 LSTM 模型：
- 读取 final_training_panel_with_rate.parquet
- 构造与 XGB / Logistic / Boost 一致的特征：
    * 原始价量/宏观/Fed
    * rolling: price_mean/min/max_10/20, ret_mean/std/cum_10/20
    * EMA: close_ema_5/10/20
    * 行业 / 规模 one-hot
- 标签：相对 60 天收益（同一天截面上，高于中位数记为 1）
- 时间滚动切分：
    * 2025Q1: train <= 2024-12-31, val: 2025-01-01 ~ 2025-03-31
    * 2025Q2: train <= 2025-03-31, val: 2025-04-01 ~ 2025-06-30
- LSTM 用长度 SEQ_LEN=15 的时间窗口；序列最后一天对应的 y_rel 作为标签
"""

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# 全局随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==============================
# 1. 构造特征 & 相对收益 label
# ==============================
def build_feature_df():
    BASE_DIR = Path("/Users/wyhmac/Desktop/SW")
    PANEL_PATH = BASE_DIR / "final_training_panel_with_rate.parquet"

    print(">>> loading panel from", PANEL_PATH)
    df = pd.read_parquet(PANEL_PATH)
    print("raw panel shape:", df.shape)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 只保留有 forward return 的样本
    if "fwd_ret_60d" not in df.columns:
        raise ValueError("fwd_ret_60d column not found in panel.")
    df = df[~df["fwd_ret_60d"].isna()].copy()

    # 日收益
    if "ret" not in df.columns:
        df["ret"] = df.groupby("ticker")["close"].pct_change()

    # === 每只股票 rolling 特征 ===
    def add_roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()

        for W in [10, 20]:
            g[f"price_mean_{W}"] = g["close"].rolling(W, min_periods=5).mean()
            g[f"price_min_{W}"] = g["close"].rolling(W, min_periods=5).min()
            g[f"price_max_{W}"] = g["close"].rolling(W, min_periods=5).max()

            g[f"ret_mean_{W}"] = g["ret"].rolling(W, min_periods=5).mean()
            g[f"ret_std_{W}"] = g["ret"].rolling(W, min_periods=5).std()

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

    # DeprecationWarning 无伤大雅
    df = df.groupby("ticker", group_keys=False).apply(add_roll)

    # === 相对收益 label：同一天截面上，相对于中位数 ===
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

    # 行业 / 规模 -> category
    for col in ["sector", "cap_bucket"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # one-hot
    df = pd.get_dummies(
        df,
        columns=[c for c in ["sector", "cap_bucket"] if c in df.columns],
        prefix=["sector", "cap"],
        drop_first=False,
    )

    # 数值特征
    base_num_cols = [
        "close", "volume", "ma15", "ret",
        "gold_price", "oil_price", "usd_index",
        "fed_sent_pos", "fed_sent_neg", "fed_sent_neu", "fed_sent_net",
        "fed_rate_hike", "fed_rate_cut", "fed_rate_hold",
        "fed_rate_decision_day", "fed_rate_change_bp",
        "corr_gold", "corr_oil", "corr_usd", "corr_fed",
    ]
    base_num_cols = [c for c in base_num_cols if c in df.columns]

    cat_cols = [c for c in df.columns if c.startswith("sector_") or c.startswith("cap_")]

    feature_cols = base_num_cols + roll_cols + cat_cols

    # 先把特征全转成数值（防止有奇怪的 object）
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # 丢掉缺失
    df = df.dropna(subset=feature_cols + ["y_rel"])

    print("with features shape:", df.shape)
    print("num features:", len(feature_cols))

    return df, feature_cols


# ==============================
# 2. 构造时间序列样本（LSTM 用）
# ==============================
def build_sequences(df: pd.DataFrame, feature_cols, seq_len: int = 15):
    """
    以每只股票为单位构造长度为 seq_len 的序列：
    - X_seq: [t-seq_len+1, ..., t] 的特征
    - y_seq: t 时刻的 y_rel
    只保留最后日期 <= 2025-06-30 的样本（因为我们只验证到 2025Q2）
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    max_date = np.datetime64("2025-06-30")

    X_list = []
    y_list = []
    d_list = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        feat = g[feature_cols].values
        y = g["y_rel"].values.astype(int)
        dates = g["date"].values.astype("datetime64[ns]")

        T = len(g)
        if T < seq_len:
            continue

        for t in range(seq_len - 1, T):
            dt = dates[t]
            if dt > max_date:
                continue

            X_seq = feat[t - seq_len + 1 : t + 1]
            y_t = y[t]

            X_list.append(X_seq)
            y_list.append(y_t)
            d_list.append(dt)

    X = np.stack(X_list, axis=0)  # (N, L, F)
    y = np.array(y_list, dtype=np.int64)
    dates = np.array(d_list, dtype="datetime64[ns]")

    print("built sequences: X shape", X.shape, " y shape", y.shape)
    return X, y, dates


def split_by_time(X, y, dates):
    """
    按最后一天的日期做时间切分：
    - 2025Q1: train <= 2024-12-31, val: 2025-01-01 ~ 2025-03-31
    - 2025Q2: train <= 2025-03-31, val: 2025-04-01 ~ 2025-06-30
    """
    d = dates.astype("datetime64[ns]")

    cut_q1_train = np.datetime64("2024-12-31")
    cut_q1_val_end = np.datetime64("2025-03-31")
    cut_q2_train = cut_q1_val_end
    cut_q2_val_end = np.datetime64("2025-06-30")

    q1_train_mask = d <= cut_q1_train
    q1_val_mask = (d > cut_q1_train) & (d <= cut_q1_val_end)

    q2_train_mask = d <= cut_q2_train
    q2_val_mask = (d > cut_q2_train) & (d <= cut_q2_val_end)

    X_q1_train, y_q1_train = X[q1_train_mask], y[q1_train_mask]
    X_q1_val, y_q1_val = X[q1_val_mask], y[q1_val_mask]

    X_q2_train, y_q2_train = X[q2_train_mask], y[q2_train_mask]
    X_q2_val, y_q2_val = X[q2_val_mask], y[q2_val_mask]

    print("Q1 train seqs:", len(y_q1_train), "val seqs:", len(y_q1_val))
    print("Q2 train seqs:", len(y_q2_train), "val seqs:", len(y_q2_val))

    return (
        X_q1_train, y_q1_train, X_q1_val, y_q1_val,
        X_q2_train, y_q2_train, X_q2_val, y_q2_val,
    )


def standardize_train_val(X_train, X_val):
    """
    用 train 的均值/方差标准化特征：
    X: (N, L, F)
    这里显式转 float64，避免 numpy 对 object 数组 std 出错。
    """
    F = X_train.shape[-1]

    # 展平后转成 float64
    flat_train = X_train.reshape(-1, F).astype("float64")
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std[std == 0] = 1.0

    flat_train_std = (flat_train - mean) / std
    flat_val = X_val.reshape(-1, F).astype("float64")
    flat_val_std = (flat_val - mean) / std

    X_train_std = flat_train_std.reshape(X_train.shape)
    X_val_std = flat_val_std.reshape(X_val.shape)

    return X_train_std, X_val_std


# ==============================
# 3. PyTorch Dataset / Model
# ==============================
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, L, F)
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]  # (B, H)
        logits = self.fc(h_last).squeeze(-1)  # (B,)
        return logits


# ==============================
# 4. 训练 & 验证
# ==============================
def train_lstm_stage(stage_name, X_train, y_train, X_val, y_val, num_features,
                     seq_len=15, epochs=5, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== LSTM {stage_name} =====")
    print("device:", device)
    print("train seqs:", len(y_train), "val seqs:", len(y_val))

    # 标准化
    X_train_std, X_val_std = standardize_train_val(X_train, X_val)

    train_ds = SeqDataset(X_train_std, y_train)
    val_ds = SeqDataset(X_val_std, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMClassifier(input_dim=num_features, hidden_dim=64, num_layers=1, dropout=0.1).to(device)

    # 类别不平衡: pos_weight = n_neg / n_pos
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    if pos == 0 or neg == 0:
        print("Warning: only one class in training labels, pos_weight disabled.")
        criterion = nn.BCEWithLogitsLoss()
    else:
        pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 训练若干 epoch
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # 简单 val 指标（阈值 0.5）
        model.eval()
        all_probs = []
        all_true = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_true.append(yb.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_true = np.concatenate(all_true)

        pred = (all_probs >= 0.5).astype(int)

        acc = accuracy_score(all_true, pred)
        f1 = f1_score(all_true, pred)
        try:
            auc = roc_auc_score(all_true, all_probs)
        except ValueError:
            auc = float("nan")
        bal = balanced_accuracy_score(all_true, pred)

        print(
            f"{stage_name} epoch {epoch} - "
            f"train_loss: {avg_loss:.4f}, "
            f"val_acc@0.5={acc:.4f}, val_f1@0.5={f1:.4f}, "
            f"val_auc={auc:.4f}, val_bal_acc={bal:.4f}"
        )

    # 最终一轮完整评估
    model.eval()
    all_probs = []
    all_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_true = np.concatenate(all_true)
    pred = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_true, pred)
    f1 = f1_score(all_true, pred)
    try:
        auc = roc_auc_score(all_true, all_probs)
    except ValueError:
        auc = float("nan")
    bal = balanced_accuracy_score(all_true, pred)

    print(f"\n===== LSTM {stage_name} FINAL @0.5 =====")
    print(f"LSTM {stage_name} @0.5 -> acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}, bal_acc={bal:.4f}")
    print("confusion_matrix:")
    print(confusion_matrix(all_true, pred))
    print("classification_report:")
    print(classification_report(all_true, pred, digits=4))


def main():
    # 1) 特征 + label + date
    df, feature_cols = build_feature_df()

    # 2) 构造序列样本（LSTM）
    SEQ_LEN = 15
    X, y, dates = build_sequences(df, feature_cols, seq_len=SEQ_LEN)

    # 3) 时间切分
    (
        X_q1_train, y_q1_train, X_q1_val, y_q1_val,
        X_q2_train, y_q2_train, X_q2_val, y_q2_val,
    ) = split_by_time(X, y, dates)

    num_features = X.shape[-1]

    # 4) 训练两个时间段的 LSTM
    train_lstm_stage("2025Q1", X_q1_train, y_q1_train, X_q1_val, y_q1_val, num_features, seq_len=SEQ_LEN)
    train_lstm_stage("2025Q2", X_q2_train, y_q2_train, X_q2_val, y_q2_val, num_features, seq_len=SEQ_LEN)


if __name__ == "__main__":
    main()
