import pandas as pd
from pathlib import Path

print(">>> running training_model_prep.py (v3: numeric coerce + safe pct_change)")

# ========== 1. 路径配置 ==========
BASE_DIR = Path("/Users/wyhmac/Desktop/SW")
STOCK_DIR = BASE_DIR / "到20251001"
MACRO_DIR = BASE_DIR / "macro_csv"
FED_DIR   = BASE_DIR / "美联储信息2"

stock_files = {
    "cd": STOCK_DIR / "cd100_20240630_20251001.xlsx",
    "energy": STOCK_DIR / "Energy100_20240630_20251001.xlsx",
    "financials": STOCK_DIR / "Financials100_20240630_20251001.xlsx",
    "industrials": STOCK_DIR / "Industrials100_20240630_20251001.xlsx",
    "it": STOCK_DIR / "IT100_20240630_20251001.xlsx",
}

gold_path = MACRO_DIR / "gold.csv"
oil_path  = MACRO_DIR / "oil.csv"
usd_path  = MACRO_DIR / "usd.csv"

fed_path  = FED_DIR / "fed_news_20240630_20251001_with_content.csv"

# ========== 2. 路径检查 ==========
print("=== checking files ===")
to_check = list(stock_files.values()) + [gold_path, oil_path, usd_path, fed_path]
missing = []
for p in to_check:
    if p.exists():
        print("✅ 找到了：", p)
    else:
        print("❌ 找不到文件：", p)
        missing.append(str(p))
print("=== check done ===")

if missing:
    raise FileNotFoundError("下面这些文件没找到，请检查路径/文件名：\n" + "\n".join(missing))

# ========== 3. 辅助函数 ==========
def cap_from_idx(idx: int) -> str:
    # 0-29 -> large, 30-64 -> mid, 65-99 -> small
    if idx < 30:
        return "large"
    elif idx < 30 + 35:
        return "mid"
    else:
        return "small"

# ========== 4. 读取5个excel × 100个sheet，展开 ==========
all_stocks = []

for sector, path in stock_files.items():
    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names
    print(f"[{sector}] 一共 {len(sheet_names)} 个sheet")
    for i, sheet_name in enumerate(sheet_names):
        cap_bucket = cap_from_idx(i)
        ticker = sheet_name.strip()

        df = pd.read_excel(path, sheet_name=sheet_name)

        # 统一列名
        df = df.rename(columns={
            "Date": "date",
            "最新价格": "close",
            "成交量": "volume",
            "移动平均 (15)": "ma15",
        })

        # 转日期
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # 👇 关键：把数值列都强制转成数值，转不了的变 NaN
        for col in ["close", "volume", "ma15"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 补上维度
        df["ticker"] = ticker
        df["sector"] = sector
        df["cap_bucket"] = cap_bucket

        all_stocks.append(df)

stocks_df = pd.concat(all_stocks, ignore_index=True)
stocks_df = stocks_df.sort_values(["ticker", "date"]).reset_index(drop=True)

print("stocks_df shape:", stocks_df.shape)
print(stocks_df.head())

# ========== 5. 宏观合并 ==========
gold = pd.read_csv(gold_path)
oil  = pd.read_csv(oil_path)
usd  = pd.read_csv(usd_path)

for m in (gold, oil, usd):
    m["date"] = pd.to_datetime(m["date"]).dt.date

macro_df = (
    gold.rename(columns={"value": "gold_price"})
        .merge(oil.rename(columns={"value": "oil_price"}), on="date", how="outer")
        .merge(usd.rename(columns={"value": "usd_index"}), on="date", how="outer")
        .sort_values("date")
        .ffill()
)

print("macro_df range:", macro_df["date"].min(), "→", macro_df["date"].max())

# ========== 6. 读取美联储新闻（published_utc 优先） ==========
fed = pd.read_csv(fed_path)
fed_cols = list(fed.columns)
print("fed columns:", fed_cols)

# 6.1 找日期列，优先: published/publish/utc -> date -> time -> 第一列
date_col = None
for c in fed.columns:
    lc = c.lower()
    if "publish" in lc or "utc" in lc:
        date_col = c
        break
if date_col is None:
    for c in fed.columns:
        if "date" in c.lower():
            date_col = c
            break
if date_col is None:
    for c in fed.columns:
        if "time" in c.lower():
            date_col = c
            break
if date_col is None:
    date_col = fed.columns[0]

print("使用这一列作为日期列:", date_col)

fed[date_col] = pd.to_datetime(fed[date_col], errors="coerce").dt.date
fed = fed.dropna(subset=[date_col])

# 6.2 找内容列
content_col = None
for c in fed.columns:
    if "content" in c.lower():
        content_col = c
        break
if content_col is None:
    for c in fed.columns:
        if "summary" in c.lower() or "text" in c.lower() or "body" in c.lower():
            content_col = c
            break
if content_col is None:
    content_col = fed.columns[-1]

print("使用这一列作为内容列:", content_col)

fed_daily_text = (
    fed.groupby(date_col)[content_col]
       .apply(lambda x: "\n\n".join([str(t) for t in x if pd.notnull(t)]))
       .reset_index()
       .rename(columns={date_col: "date", content_col: "fed_text"})
)

print("fed_daily_text rows:", len(fed_daily_text))

# ========== 7. 合并：股票 + 宏观 + 美联储 ==========
full = (
    stocks_df
    .merge(macro_df, on="date", how="left")
    .merge(fed_daily_text, on="date", how="left")
    .sort_values(["ticker", "date"])
    .reset_index(drop=True)
)

# 宏观从 2024-07-01 开始，裁掉更早的
full = full[full["date"] >= pd.to_datetime("2024-07-01").date()].reset_index(drop=True)

# 👇 再保险一遍：合并后也把 close/volume 转成数值
for col in ["close", "volume", "ma15", "gold_price", "oil_price", "usd_index"]:
    if col in full.columns:
        full[col] = pd.to_numeric(full[col], errors="coerce")

# 丢掉没有价格的行（不能算收益）
full = full.dropna(subset=["close"]).reset_index(drop=True)

# ========== 8. 做收益 & 前瞻收益（安全版） ==========
# pct_change 这里指定 fill_method=None，可以去掉那个 FutureWarning
full["ret"] = (
    full.sort_values(["ticker", "date"])
        .groupby("ticker")["close"]
        .pct_change(fill_method=None)
)

HORIZON = 60
full["fwd_ret_60d"] = (
    full.sort_values(["ticker", "date"])
        .groupby("ticker")["close"]
        .shift(-HORIZON) / full["close"] - 1.0
)

print("final merged shape:", full.shape)
print(full.head(15))

# ========== 9. 保存 ==========
out_path = BASE_DIR / "merged_panel_for_lstm.parquet"
full.to_parquet(out_path, index=False)
print("✅ 已保存到：", out_path)
