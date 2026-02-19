#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_finbert_sentiment.py

从美联储新闻 CSV 中提取文本，用 FinBERT 计算情绪（pos/neg/neu），
再按日期聚合成日度情绪因子 fed_sent_pos/neg/neu/net。

运行方式（建议先创建 venv 并安装 transformers/torch）:
    /usr/local/bin/python3 "/Users/wyhmac/Desktop/SW/train_finbert_sentiment.py"
"""

import os
from pathlib import Path
import math
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ===================== 路径配置 =====================
# !!! 如需调整路径，只改这里即可
BASE_DIR = Path("/Users/wyhmac/Desktop/SW")
FED_NEWS_CSV = BASE_DIR / "美联储信息2" / "fed_news_20240630_20251001_with_content.csv"

ARTICLES_OUT = BASE_DIR / "fed_sentiment_articles.csv"
DAILY_OUT    = BASE_DIR / "fed_sentiment_daily.csv"

# 使用的 FinBERT 模型（HuggingFace id）
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"


# ===================== 工具函数 =====================
def chunked(iterable, batch_size):
    """按 batch_size 切分列表，避免一次性送入 GPU/CPU 内存过大。"""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def load_fed_news(csv_path: Path) -> pd.DataFrame:
    """读取美联储新闻 CSV 并做最基本清洗。"""
    print(f">>> loading Fed news from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 只保留确实有内容的行
    if "content" not in df.columns:
        raise ValueError("Input CSV must have a 'content' column with article text.")

    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"] != ""].copy()

    # 解析时间，生成 date 列（只用日期不含具体时间）
    if "published_utc" not in df.columns:
        raise ValueError("Input CSV must have a 'published_utc' column.")

    df["published_dt"] = pd.to_datetime(df["published_utc"], errors="coerce")
    df = df.dropna(subset=["published_dt"]).copy()
    df["date"] = df["published_dt"].dt.date

    print(f"loaded {len(df)} articles with non-empty content and valid timestamp.")
    return df


def load_finbert(model_name: str = FINBERT_MODEL_NAME):
    """加载 FinBERT 模型与 tokenizer。"""
    print(f">>> loading FinBERT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"model device: {device}")
    return tokenizer, model, device


def infer_sentiment(df: pd.DataFrame,
                    tokenizer,
                    model,
                    device,
                    text_col: str = "content",
                    batch_size: int = 16) -> pd.DataFrame:
    """
    使用 FinBERT 对 df[text_col] 进行情绪预测，返回带有
    sent_pos / sent_neg / sent_neu / sent_net 列的 DataFrame。
    """
    texts = df[text_col].tolist()
    n = len(texts)
    print(f">>> running FinBERT sentiment on {n} documents ...")

    # 从模型配置中读出 label -> id 映射，避免硬编码顺序
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_to_idx = {v.lower(): k for k, v in id2label.items()}

    # 兼容不同大小写/命名
    pos_idx = label_to_idx.get("positive")
    neg_idx = label_to_idx.get("negative")
    neu_idx = label_to_idx.get("neutral")

    if pos_idx is None or neg_idx is None or neu_idx is None:
        raise RuntimeError(f"Unexpected FinBERT labels: {id2label}. "
                           f"Expecting 'positive', 'negative', 'neutral'.")

    all_pos = []
    all_neg = []
    all_neu = []

    with torch.no_grad():
        for batch in chunked(texts, batch_size):
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,  # 截断超长文本
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()

            all_pos.extend(probs[:, pos_idx].tolist())
            all_neg.extend(probs[:, neg_idx].tolist())
            all_neu.extend(probs[:, neu_idx].tolist())

    df = df.copy()
    df["sent_pos"] = all_pos
    df["sent_neg"] = all_neg
    df["sent_neu"] = all_neu
    df["sent_net"] = df["sent_pos"] - df["sent_neg"]

    print(">>> finished sentiment inference.")
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 date 聚合情绪，生成 fed_sent_pos/neg/neu/net 与文章数 n_articles。
    """
    print(">>> aggregating to daily sentiment factors ...")
    grp = df.groupby("date")

    daily = grp.agg(
        fed_sent_pos=("sent_pos", "mean"),
        fed_sent_neg=("sent_neg", "mean"),
        fed_sent_neu=("sent_neu", "mean"),
        fed_sent_net=("sent_net", "mean"),
        n_articles=("sent_net", "size"),
    ).reset_index()

    # 排序一下日期
    daily = daily.sort_values("date").reset_index(drop=True)
    print(f"generated {len(daily)} daily sentiment rows.")
    return daily


def main():
    # 1. 读入新闻
    df_news = load_fed_news(FED_NEWS_CSV)

    # 2. 加载 FinBERT
    tokenizer, model, device = load_finbert(FINBERT_MODEL_NAME)

    # 3. 逐篇文章打分
    df_sent = infer_sentiment(df_news, tokenizer, model, device)

    # 4. 保存逐篇文章结果（方便 debug / 以后做更细粒度分析）
    cols_to_save = [
        "source", "feed", "title", "link",
        "published_utc", "published_dt", "date",
        "sent_pos", "sent_neg", "sent_neu", "sent_net",
    ]
    # 某些列可能不存在，就过滤一下
    cols_to_save = [c for c in cols_to_save if c in df_sent.columns]

    print(f">>> saving article-level sentiment to: {ARTICLES_OUT}")
    df_sent.to_csv(ARTICLES_OUT, index=False, columns=cols_to_save)

    # 5. 按日期聚合为 fed_sent_* 因子
    df_daily = aggregate_daily(df_sent)

    print(f">>> saving daily sentiment factors to: {DAILY_OUT}")
    df_daily.to_csv(DAILY_OUT, index=False)

    print("✅ done: FinBERT sentiment (articles + daily factors) generated.")


if __name__ == "__main__":
    main()
