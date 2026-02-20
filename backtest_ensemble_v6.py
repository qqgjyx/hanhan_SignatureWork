import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# =========================
# 基本路径
# =========================
BASE_DIR = Path("/Users/wyhmac/Desktop/SW")
PANEL_PATH = BASE_DIR / "final_training_panel_with_rate.parquet"


# =========================
# 1. 读入面板 + 基础清洗
# =========================
def load_panel():
    print(">>> loading panel ...")
    df = pd.read_parquet(PANEL_PATH)

    # 日期转为 date 类型
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # 一些关键列转成数值，防止混进字符串
    numeric_cols = [
        "close",
        "volume",
        "ma15",
        "gold_price",
        "oil_price",
        "usd_index",
        "ret",
        "fwd_ret_60d",
        "fed_sent_pos",
        "fed_sent_neg",
        "fed_sent_neu",
        "fed_sent_net",
        "corr_gold",
        "corr_oil",
        "corr_usd",
        "corr_fed",
        "fed_rate_hike",
        "fed_rate_cut",
        "fed_rate_hold",
        "fed_rate_change_bp",
        "fed_rate_decision_day",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 按股票 & 日期排序
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 只保留有 60 天前瞻收益的部分，避免 label 缺失
    df = df[~df["fwd_ret_60d"].isna()].copy()

    print("panel shape:", df.shape)
    return df


# =========================
# 2. 特征工程（和 v6 同风格）
# =========================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始面板上加：
    - 各种 rolling price/return 特征
    - EMA
    - 行业 / 规模 的相对表现
    - 行业 / 规模 one-hot + 和 fed_sent_net 的交互项
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])
    g = df.groupby("ticker")

    # --- rolling on close / ret ---
    for win in [5, 10, 20]:
        df[f"price_mean_{win}"] = g["close"].transform(
            lambda x: x.rolling(win, min_periods=3).mean()
        )
        df[f"price_min_{win}"] = g["close"].transform(
            lambda x: x.rolling(win, min_periods=3).min()
        )
        df[f"price_max_{win}"] = g["close"].transform(
            lambda x: x.rolling(win, min_periods=3).max()
        )

        df[f"ret_mean_{win}"] = g["ret"].transform(
            lambda x: x.rolling(win, min_periods=3).mean()
        )
        df[f"ret_std_{win}"] = g["ret"].transform(
            lambda x: x.rolling(win, min_periods=3).std()
        )
        df[f"ret_cum_{win}"] = g["ret"].transform(
            lambda x: x.rolling(win, min_periods=3).sum()
        )

    # --- EMA ---
    for span in [5, 10, 20]:
        df[f"close_ema_{span}"] = g["close"].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )

    # --- 行业 / 规模的相对表现 (20 日窗口) ---
    df["date_ts"] = pd.to_datetime(df["date"])

    # 行业 20 日平均收益
    df = df.sort_values(["sector", "date_ts"])
    g_sec = df.groupby("sector")
    df["ret20_sector_mean"] = g_sec["ret"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

    # 规模 20 日平均收益
    df = df.sort_values(["cap_bucket", "date_ts"])
    g_cap = df.groupby("cap_bucket")
    df["ret20_cap_mean"] = g_cap["ret"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

    # alpha_vs_cap_20：相对同规模组的超额收益
    df = df.sort_values(["ticker", "date_ts"])
    g_t = df.groupby("ticker")

    def alpha_cap_20(grp):
        diff = grp["ret"] - grp["ret20_cap_mean"]
        return diff.rolling(20, min_periods=5).mean()

    df["alpha_vs_cap_20"] = g_t.apply(alpha_cap_20).reset_index(level=0, drop=True)

    # 删除临时的 timestamp 列
    df = df.drop(columns=["date_ts"])

    # --- 行业 / 规模 one-hot ---
    df = pd.get_dummies(df, columns=["sector", "cap_bucket"], prefix=["sector", "cap"])

    # --- Fed 情绪 × 行业/规模 交互 ---
    if "fed_sent_net" in df.columns:
        for col in df.columns:
            if col.startswith("sector_") or col.startswith("cap_"):
                df[col + "__x__fed_sent_net"] = df[col] * df["fed_sent_net"]

    return df


# =========================
# 3. 构造 y、选特征
# =========================
def build_dataset_with_features(df: pd.DataFrame):
    # 加特征
    df = add_features(df)

    # 构造二分类标签：未来 60 天涨跌
    df["target"] = (df["fwd_ret_60d"] > 0).astype(int)

    # 只保留从 2024-07-01 开始的数据（和之前版本一致）
    df = df[df["date"] >= datetime.date(2024, 7, 1)].copy()

    # 数值特征列：所有 numeric，但去掉泄露列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for leak in ["fwd_ret_60d", "target"]:
        if leak in num_cols:
            num_cols.remove(leak)
    feature_cols = sorted(num_cols)

    # 丢掉特征或标签有 NaN 的行
    df = df.dropna(subset=feature_cols + ["target", "fwd_ret_60d"]).reset_index(
        drop=True
    )

    print("with features shape:", df.shape)
    print("num features:", len(feature_cols))
    return df, feature_cols


# =========================
# 4. 训练 + 误差加权再拟合的 ensemble
# =========================
def fit_ensemble_error_refit(df_train: pd.DataFrame, feature_cols):
    X_tr = df_train[feature_cols].values
    y_tr = df_train["target"].values

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)

    # ===== Base 模型 =====
    lr = LogisticRegression(
        max_iter=200, class_weight="balanced", n_jobs=-1, solver="lbfgs"
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1),
        random_state=42,
        n_jobs=-1,
    )

    lr.fit(X_tr_sc, y_tr)
    rf.fit(X_tr, y_tr)
    xgb.fit(X_tr, y_tr)

    def predict_base_proba(X):
        X_sc = scaler.transform(X)
        p1 = lr.predict_proba(X_sc)[:, 1]
        p2 = rf.predict_proba(X)[:, 1]
        p3 = xgb.predict_proba(X)[:, 1]
        return (p1 + p2 + p3) / 3.0

    # ===== 用 base 的误差做 sample_weight，再拟合一轮 =====
    p_tr_base = predict_base_proba(X_tr)
    w = 1.0 + 2.0 * np.abs(y_tr - p_tr_base)

    lr2 = LogisticRegression(
        max_iter=200, class_weight="balanced", n_jobs=-1, solver="lbfgs"
    )
    rf2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=123,
        n_jobs=-1,
    )
    xgb2 = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1),
        random_state=123,
        n_jobs=-1,
    )

    lr2.fit(X_tr_sc, y_tr, sample_weight=w)
    rf2.fit(X_tr, y_tr, sample_weight=w)
    xgb2.fit(X_tr, y_tr, sample_weight=w)

    def predict_proba(X):
        X_sc = scaler.transform(X)
        p1 = lr2.predict_proba(X_sc)[:, 1]
        p2 = rf2.predict_proba(X)[:, 1]
        p3 = xgb2.predict_proba(X)[:, 1]
        return (p1 + p2 + p3) / 3.0

    models = {
        "scaler": scaler,
        "lr": lr,
        "rf": rf,
        "xgb": xgb,
        "lr2": lr2,
        "rf2": rf2,
        "xgb2": xgb2,
    }
    return models, predict_proba


# =========================
# 5. 简单分类评估（看一下，不当作真正回测）
# =========================
def eval_classification(y_true, p_hat, name=""):
    y_pred_05 = (p_hat >= 0.5).astype(int)
    auc = roc_auc_score(y_true, p_hat)
    acc = accuracy_score(y_true, y_pred_05)
    bal_acc = balanced_accuracy_score(y_true, y_pred_05)
    f1 = f1_score(y_true, y_pred_05)
    print(f"{name} AUC={auc:.4f}  acc@0.5={acc:.4f}  bal_acc@0.5={bal_acc:.4f}  f1@0.5={f1:.4f}")


# =========================
# 6. 回测：每天选 top 20% 概率做多，比较 60d 未来收益
# =========================
def backtest_stage(df_val: pd.DataFrame, p_val: np.ndarray, stage_name: str):
    tmp = df_val.copy()
    tmp["p_up"] = p_val

    # 只保留有 60d 未来收益的样本
    tmp = tmp.dropna(subset=["fwd_ret_60d", "p_up"])

    records = []
    for d, g in tmp.groupby("date"):
        if len(g) < 20:
            continue
        # 每天选 top 20% 概率做多
        thr = g["p_up"].quantile(0.8)
        long = g[g["p_up"] >= thr]
        rest = g[g["p_up"] < thr]

        if long.empty or rest.empty:
            continue

        records.append(
            {
                "date": d,
                "n_long": len(long),
                "avg_ret_long": long["fwd_ret_60d"].mean(),
                "avg_ret_all": g["fwd_ret_60d"].mean(),
                "avg_ret_rest": rest["fwd_ret_60d"].mean(),
            }
        )

    daily = pd.DataFrame(records).sort_values("date")

    print(f"\n=== Backtest {stage_name} (top 20% each day, 60d forward return) ===")
    print("days with portfolio:", len(daily))
    if len(daily) > 0:
        print("mean 60d return LONG:", daily["avg_ret_long"].mean())
        print("mean 60d return REST:", daily["avg_ret_rest"].mean())
        print("mean 60d return ALL: ", daily["avg_ret_all"].mean())
    else:
        print("no valid days (maybe all NaN?)")

    out_path = BASE_DIR / f"backtest_{stage_name}_daily.csv"
    daily.to_csv(out_path, index=False)
    print("saved daily backtest to:", out_path)

    return daily


# =========================
# 7. 一个阶段的完整流程：训练 + 预测 + 回测
# =========================
def run_stage(df, feature_cols, train_end, val_start, val_end, stage_name):
    print(
        f"\n===== Stage {stage_name} (train <= {train_end} | val {val_start} ~ {val_end}) ====="
    )
    train_mask = df["date"] <= train_end
    val_mask = (df["date"] >= val_start) & (df["date"] <= val_end)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    print("train size:", len(df_train), "val size:", len(df_val))

    models, predict_proba = fit_ensemble_error_refit(df_train, feature_cols)

    X_val = df_val[feature_cols].values
    y_val = df_val["target"].values
    p_val = predict_proba(X_val)

    eval_classification(y_val, p_val, name=f"{stage_name} (error-refit ensemble)")

    # 做简单选股回测
    backtest_stage(df_val, p_val, stage_name)


# =========================
# 8. 主流程
# =========================
def main():
    df = load_panel()
    df_feat, feature_cols = build_dataset_with_features(df)

    # 两个时间段
    stage1_train_end = datetime.date(2024, 12, 31)
    stage1_val_start = datetime.date(2025, 1, 1)
    stage1_val_end = datetime.date(2025, 3, 31)

    stage2_train_end = datetime.date(2025, 3, 31)
    stage2_val_start = datetime.date(2025, 4, 1)
    stage2_val_end = datetime.date(2025, 6, 30)

    run_stage(
        df_feat,
        feature_cols,
        train_end=stage1_train_end,
        val_start=stage1_val_start,
        val_end=stage1_val_end,
        stage_name="2025Q1",
    )

    run_stage(
        df_feat,
        feature_cols,
        train_end=stage2_train_end,
        val_start=stage2_val_start,
        val_end=stage2_val_end,
        stage_name="2025Q2",
    )

    print("\n✅ backtest done.")


if __name__ == "__main__":
    main()
