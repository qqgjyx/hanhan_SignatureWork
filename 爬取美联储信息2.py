# -*- coding: utf-8 -*-
"""
美联储（Board of Governors）新闻信息一键抓取脚本（历史区间稳健版）
- 时间窗：2024-06-30 ~ 2025-10-01（含）
- 源：RSS（尽力） + 年度归档（Press / Speeches / Testimony）
- 日期解析：优先 URL 中的 YYYYMMDD；其次 "Month DD, YYYY"
- 去重 / 重试 / 节流；输出 CSV + JSON 到 /Users/wyhmac/Desktop/SW/美联储信息2
"""

import csv, json, re, time, sys, traceback
from datetime import datetime, timezone
from urllib.parse import urljoin
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import feedparser
from requests.adapters import HTTPAdapter, Retry

import socket
import urllib3.util.connection as urllib3_cn

def _allowed_gai_family():
    # 只用 IPv4，避免 IPv6 网络不稳定导致的握手/读取超时
    return socket.AF_INET

urllib3_cn.allowed_gai_family = _allowed_gai_family


# ===== 固定参数 =====
START_DATE = "2024-06-30"
END_DATE   = "2025-10-01"

OUT_DIR     = Path("/Users/wyhmac/Desktop/SW/美联储信息2")
OUTPUT_CSV  = OUT_DIR / "fed_news_20240630_20251001.csv"
OUTPUT_JSON = OUT_DIR / "fed_news_20240630_20251001.json"

FEEDS_INDEX_URL = "https://www.federalreserve.gov/feeds/feeds.htm"
ARCHIVE_BASES = {
    "press":     "https://www.federalreserve.gov/newsevents/pressreleases/{y}-press.htm",
    "speeches":  "https://www.federalreserve.gov/newsevents/speech/{y}-speech.htm",
    "testimony": "https://www.federalreserve.gov/newsevents/testimony/{y}-testimony.htm",
}

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
REQUEST_TIMEOUT   = 20
REQUEST_SLEEP_SEC = 0.25
RETRY_TOTAL       = 3
RETRY_BACKOFF     = 0.6

def banner():
    print("="*84, flush=True)
    print("Fed News Fetch — FULL VERSION".encode('utf-8').decode('utf-8'), flush=True)
    print(f"Window  : {START_DATE}  ->  {END_DATE}", flush=True)
    print(f"Outputs :\n  CSV  = {OUTPUT_CSV}\n  JSON = {OUTPUT_JSON}", flush=True)
    print("="*84, flush=True)

def build_session():
    s = requests.Session()
    retries = Retry(
        total=RETRY_TOTAL, backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": UA})
    return s

def to_utc_day(dt_str, end=False):
    dt = datetime.strptime(dt_str, "%Y-%m-%d")
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.replace(tzinfo=timezone.utc)

START_UTC = to_utc_day(START_DATE)
END_UTC   = to_utc_day(END_DATE, end=True)

def within(dt_utc: datetime) -> bool:
    return START_UTC <= dt_utc <= END_UTC

# ---- 日期解析 ----
def parse_date_from_url(url: str):
    # 连续8位（YYYYMMDD）
    m = re.search(r"(?<!\d)(\d{8})(?!\d)", url)
    if m:
        y, mo, d = m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:8]
        try:
            return datetime(int(y), int(mo), int(d), tzinfo=timezone.utc)
        except Exception:
            pass
    # 形如 YYYY-MM-DD / YYYY_MM_DD
    m2 = re.search(r"(\d{4})[-_/](\d{2})[-_/](\d{2})", url)
    if m2:
        try:
            return datetime(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)), tzinfo=timezone.utc)
        except Exception:
            pass
    return None

def parse_date_from_text(text: str):
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        text
    )
    if m:
        try:
            return datetime.strptime(m.group(0), "%B %d, %Y").replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None

# ---- RSS ----
NEWS_KEYWORDS = ["press", "speech", "testimony", "speeches_and_testimony"]

def normalize_feed_url(href: str, base: str) -> str | None:
    if not href or not href.strip().lower().endswith(".xml"):
        return None
    abs_url = urljoin(base, href.strip())
    if "/feeds/" not in abs_url:
        return None
    return abs_url

def discover_feed_urls(session) -> list[str]:
    urls = set()
    try:
        r = session.get(FEEDS_INDEX_URL, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            abs_url = normalize_feed_url(a["href"], FEEDS_INDEX_URL)
            if abs_url:
                urls.add(abs_url)
    except Exception as e:
        print(f"[WARN] 访问 RSS 索引失败：{e}", flush=True)

    # 已知 feed 兜底
    known = [
        "/feeds/press_all.xml",
        "/feeds/press_monetary.xml",
        "/feeds/press_bcreg.xml",
        "/feeds/press_enforcement.xml",
        "/feeds/press_orders.xml",
        "/feeds/press_other.xml",
        "/feeds/speeches.xml",
        "/feeds/speeches_and_testimony.xml",
        "/feeds/testimony.xml",
    ]
    for path in known:
        urls.add(urljoin(FEEDS_INDEX_URL, path))

    urls = sorted({u for u in urls if any(k in u.lower() for k in NEWS_KEYWORDS)})
    print(f"发现 {len(urls)} 个 RSS 源：", flush=True)
    for u in urls:
        print("  -", u, flush=True)
    return urls

def feed_datetime(entry):
    dt_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not dt_struct:
        return None
    return datetime(*dt_struct[:6], tzinfo=timezone.utc)

def fetch_from_rss(feed_urls):
    seen = set()
    rows = []
    for i, url in enumerate(feed_urls, start=1):
        print(f"[RSS {i}/{len(feed_urls)}] {url}", flush=True)
        try:
            d = feedparser.parse(url)
        except Exception as e:
            print(f"  ! feedparser 出错：{e}", flush=True)
            time.sleep(REQUEST_SLEEP_SEC)
            continue
        time.sleep(REQUEST_SLEEP_SEC)
        for e in d.entries:
            dt = feed_datetime(e)
            if not dt or not within(dt):
                continue
            link = (getattr(e, "link", "") or "").strip()
            if not link or link in seen:
                continue
            seen.add(link)
            title = (getattr(e, "title", "") or "").strip()
            summary = (getattr(e, "summary", "") or "").strip()
            categories = [t["term"] for t in getattr(e, "tags", []) if "term" in t]
            feed_title = d.feed.get("title", "").strip() if getattr(d, "feed", None) else ""
            rows.append({
                "source": "rss",
                "feed": feed_title,
                "title": title,
                "link": link,
                "published_utc": dt.isoformat(),
                "categories": ";".join(categories),
                "summary": summary
            })
    rows.sort(key=lambda x: x["published_utc"], reverse=True)
    return rows

# ---- 年度归档 ----
def fetch_one_archive_list(session, url: str, label: str):
    print(f"[归档] {label} -> {url}", flush=True)
    rows = []
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            return rows
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"  ! 抓取归档页失败：{e}", flush=True)
        return rows

    containers = soup.select("li, article, section, div")
    seen_local = set()
    for node in containers:
        a = node.find("a", href=True)
        if not a:
            continue
        href = a["href"].strip()
        link = href if href.startswith("http") else urljoin(url, href)
        if link in seen_local:
            continue

        dt = parse_date_from_url(link)
        if not dt:
            text = " ".join(node.get_text(" ").split())
            dt = parse_date_from_text(text)
        if not dt or not within(dt):
            continue

        seen_local.add(link)
        title = a.get_text(strip=True) or "(no title)"
        rows.append({
            "source": "archive",
            "feed": f"{label} (archive)",
            "title": title,
            "link": link,
            "published_utc": dt.isoformat(),
            "categories": label,
            "summary": ""
        })
    time.sleep(REQUEST_SLEEP_SEC)
    rows.sort(key=lambda x: x["published_utc"], reverse=True)
    return rows

def fetch_archives(session):
    rows = []
    start_y = to_utc_day(START_DATE).year
    end_y   = to_utc_day(END_DATE).year
    years = range(start_y - 1, end_y + 2)  # 缓冲前后一年，避免边界遗漏
    tasks = []
    for y in years:
        for label, tpl in ARCHIVE_BASES.items():
            tasks.append((label, tpl.format(y=y)))
    for label, url in tasks:
        rows.extend(fetch_one_archive_list(session, url, label))
    rows.sort(key=lambda x: x["published_utc"], reverse=True)
    return rows

# ---- 输出 ----
def save_outputs(rows, csv_path: Path, json_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source","feed","title","link","published_utc","categories","summary"])
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

# ---- 主流程 ----
def main():
    banner()
    session = build_session()

    feeds = discover_feed_urls(session)
    rows_rss = fetch_from_rss(feeds)
    if not rows_rss:
        print("\n[提示] RSS 未命中该历史区间，转用年度归档抓取三大类（Press/Speeches/Testimony）...", flush=True)

    rows_arch = fetch_archives(session)

    all_rows = []
    seen = set()
    for r in rows_rss + rows_arch:
        if r["link"] in seen:
            continue
        seen.add(r["link"])
        all_rows.append(r)

    all_rows.sort(key=lambda x: x["published_utc"], reverse=True)
    save_outputs(all_rows, OUTPUT_CSV, OUTPUT_JSON)

    print(f"\n✅ 完成，共收集 {len(all_rows)} 条新闻（区间 {START_DATE} ~ {END_DATE}）。", flush=True)
    print(f"📄 CSV:  {OUTPUT_CSV}", flush=True)
    print(f"📄 JSON: {OUTPUT_JSON}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Script crashed with exception:", flush=True)
        traceback.print_exc()
        sys.exit(1)
