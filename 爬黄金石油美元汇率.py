# -*- coding: utf-8 -*-
import pandas as pd
import yfinance as yf
from pathlib import Path

START = "2024-06-30"
END   = "2025-10-01"  # yfinance 的 end 是开区间

SOURCES = {
    # 黄金：先试期货，再试即期与 ETF
    "gold": ["GC=F", "XAUUSD=X", "GLD"],
    # 原油：先试 WTI，再试布油与 ETF
    "oil":  ["CL=F", "BZ=F", "USO"],
    # 美元：先试美元指数，再试 ETF
    "usd":  ["DX-Y.NYB", "DXY", "UUP"],
}

def fetch_one(ticker: str, start: str, end: str) -> pd.Series | None:
    print(f"-> 尝试抓取: {ticker}")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or df.empty:
            print(f"   空数据: {ticker}")
            return None
        if "Close" not in df.columns:
            print(f"   无 Close 列: {ticker}，实际列：{list(df.columns)}")
            return None
        s = df["Close"].copy()
        s.index.name = "date"
        print(f"   成功: {ticker}, 行数={len(s)}, 日期 {s.index.min().date()} ~ {s.index.max().date()}")
        return s
    except Exception as e:
        # 打印真实异常，方便定位
        print(f"   抓取 {ticker} 异常：{type(e).__name__}: {e}")
        return None

def fetch_with_fallback(names: list[str], start: str, end: str) -> pd.Series:
    for t in names:
        s = fetch_one(t, start, end)
        if s is not None:
            return s
    raise RuntimeError(f"这些 ticker 都失败了: {names}")

def save_series_as_csv(name: str, s: pd.Series, out_dir: Path):
    out = s.reset_index()
    out.columns = ["date", "value"]  # 统一成 (date,value)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.csv"
    out.to_csv(p, index=False)
    print(f"✅ 已保存: {p} 行数={len(out)}")

def main():
    out_dir = Path("./macro_csv")
    for name, tickers in SOURCES.items():
        s = fetch_with_fallback(tickers, START, END)
        save_series_as_csv(name, s, out_dir)

if __name__ == "__main__":
    main()
