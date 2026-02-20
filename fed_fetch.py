# -*- coding: utf-8 -*-
"""
Fed news fetch (RSS + yearly archives + FOMC pages) — DEBUG BUILD
Window: 2024-06-30 ~ 2025-10-01
Outputs: ./fed_news_20240630_20251001.{csv,json}
"""

import csv, json, re, time, sys, traceback
from datetime import datetime, timezone
from urllib.parse import urljoin
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import feedparser
from requests.adapters import HTTPAdapter, Retry

# ---- Force IPv4 (avoid TLS/IPv6 stalls) ----
import socket
import urllib3.util.connection as urllib3_cn
def _allowed_gai_family(): return socket.AF_INET
urllib3_cn.allowed_gai_family = _allowed_gai_family

# ---- Config ----
START_DATE = "2024-06-30"
END_DATE   = "2025-10-01"

OUT_DIR     = Path(__file__).resolve().parent
OUTPUT_CSV  = OUT_DIR / "fed_news_20240630_20251001.csv"
OUTPUT_JSON = OUT_DIR / "fed_news_20240630_20251001.json"

FEEDS_INDEX_URL = "https://www.federalreserve.gov/feeds/feeds.htm"
ARCHIVE_BASES = {
    "press":     "https://www.federalreserve.gov/newsevents/pressreleases/{y}-press.htm",
    "speeches":  "https://www.federalreserve.gov/newsevents/speech/{y}-speech.htm",
    "testimony": "https://www.federalreserve.gov/newsevents/testimony/{y}-testimony.htm",
}

# FOMC pages to scan
FOMC_PAGES = [
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes.htm",
    "https://www.federalreserve.gov/monetarypolicy/fomcstatements.htm",
]

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
REQUEST_TIMEOUT   = (6, 15)  # (connect, read)
REQUEST_SLEEP_SEC = 0.2
RETRY_TOTAL       = 2
RETRY_BACKOFF     = 0.5

def log(msg): print(msg, flush=True)

def banner():
    log("="*80)
    log("Fed News Fetch — DEBUG BUILD (with FOMC pages)")
    log(f"Window  : {START_DATE}  ->  {END_DATE}")
    log(f"Outputs :\n  CSV  = {OUTPUT_CSV}\n  JSON = {OUTPUT_JSON}")
    log("="*80)

def build_session():
    log("[STEP] build requests session ...")
    s = requests.Session()
    retries = Retry(total=RETRY_TOTAL, backoff_factor=RETRY_BACKOFF,
                    status_forcelist=[429,500,502,503,504], allowed_methods=["GET","HEAD"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": UA})
    log("[OK] session ready")
    return s

def to_utc_day(dt_str, end=False):
    dt = datetime.strptime(dt_str, "%Y-%m-%d")
    if end: dt = dt.replace(hour=23, minute=59, second=59)
    return dt.replace(tzinfo=timezone.utc)
START_UTC = to_utc_day(START_DATE); END_UTC = to_utc_day(END_DATE, end=True)
def within(dt_utc): return START_UTC <= dt_utc <= END_UTC

def parse_date_from_url(url: str):
    m = re.search(r"(?<!\d)(\d{8})(?!\d)", url)
    if m:
        y, mo, d = m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:8]
        try: return datetime(int(y), int(mo), int(d), tzinfo=timezone.utc)
        except: pass
    m2 = re.search(r"(\d{4})[-_/](\d{2})[-_/](\d{2})", url)
    if m2:
        try: return datetime(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)), tzinfo=timezone.utc)
        except: pass
    return None

def parse_date_from_text(text: str):
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", text)
    if m:
        try: return datetime.strptime(m.group(0), "%B %d, %Y").replace(tzinfo=timezone.utc)
        except: return None
    return None

def pick_date_for_anchor(a_tag):
    # 在祖先/兄弟附近找美式日期
    cur = a_tag
    for _ in range(4):  # 向上找 4 层
        txt = " ".join(cur.get_text(" ").split())
        dt = parse_date_from_text(txt)
        if dt: return dt
        if cur.parent: cur = cur.parent
        else: break
    # 向前找前一个标题
    prev = a_tag
    for _ in range(10):
        prev = prev.find_previous(["h1","h2","h3","h4","h5","h6","strong"])
        if not prev: break
        txt = " ".join(prev.get_text(" ").split())
        dt = parse_date_from_text(txt)
        if dt: return dt
    return None

# ---------------- RSS ----------------
NEWS_KEYWORDS = ["press","speech","testimony","speeches_and_testimony"]
def normalize_feed_url(href: str, base: str):
    if not href or not href.strip().lower().endswith(".xml"): return None
    abs_url = urljoin(base, href.strip())
    if "/feeds/" not in abs_url: return None
    return abs_url

def discover_feed_urls(session):
    log("[STEP] discover feed URLs ...")
    urls = set()
    try:
        log(f"  GET {FEEDS_INDEX_URL} (timeout={REQUEST_TIMEOUT})")
        r = session.get(FEEDS_INDEX_URL, timeout=REQUEST_TIMEOUT)
        log(f"  -> status {r.status_code}")
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            abs_url = normalize_feed_url(a["href"], FEEDS_INDEX_URL)
            if abs_url: urls.add(abs_url)
    except Exception as e:
        log(f"[WARN] 访问 RSS 索引失败：{e}")

    known = ["/feeds/press_all.xml","/feeds/press_monetary.xml","/feeds/press_bcreg.xml",
             "/feeds/press_enforcement.xml","/feeds/press_orders.xml","/feeds/press_other.xml",
             "/feeds/speeches.xml","/feeds/speeches_and_testimony.xml","/feeds/testimony.xml"]
    for path in known: urls.add(urljoin(FEEDS_INDEX_URL, path))

    urls = sorted({u for u in urls if any(k in u.lower() for k in NEWS_KEYWORDS)})
    log(f"[OK] 发现 {len(urls)} 个 RSS 源：")
    for u in urls: log("  - " + u)
    return urls

def feed_datetime(entry):
    dt_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not dt_struct: return None
    return datetime(*dt_struct[:6], tzinfo=timezone.utc)

def fetch_from_rss(feed_urls):
    log("[STEP] fetch from RSS ...")
    seen, rows = set(), []
    for i, url in enumerate(feed_urls, start=1):
        log(f"  [RSS {i}/{len(feed_urls)}] {url}")
        try:
            d = feedparser.parse(url)
        except Exception as e:
            log(f"    ! feedparser 出错：{e}"); time.sleep(REQUEST_SLEEP_SEC); continue
        time.sleep(REQUEST_SLEEP_SEC)
        for e in d.entries:
            dt = feed_datetime(e)
            if not dt or not within(dt): continue
            link = (getattr(e, "link", "") or "").strip()
            if not link or link in seen: continue
            seen.add(link)
            title = (getattr(e, "title", "") or "").strip()
            summary = (getattr(e, "summary", "") or "").strip()
            categories = [t["term"] for t in getattr(e, "tags", []) if "term" in t]
            feed_title = d.feed.get("title", "").strip() if getattr(d, "feed", None) else ""
            rows.append({"source":"rss","feed":feed_title,"title":title,"link":link,
                         "published_utc":dt.isoformat(),"categories":";".join(categories),"summary":summary})
    log(f"[OK] RSS 命中条目：{len(rows)}")
    rows.sort(key=lambda x: x["published_utc"], reverse=True); return rows

# ---------------- Press/Speech/Testimony archives ----------------
def fetch_one_archive_list(session, url: str, label: str):
    log(f"  [ARCHIVE] {label} -> {url}")
    rows = []
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        log(f"    -> status {r.status_code}")
        if r.status_code >= 400: return rows
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log(f"    ! 抓取归档页失败：{e}"); return rows

    containers = soup.select("li, article, section, div")
    seen_local = set()
    for node in containers:
        a = node.find("a", href=True)
        if not a: continue
        href = a["href"].strip()
        link = href if href.startswith("http") else urljoin(url, href)
        if link in seen_local: continue

        dt = parse_date_from_url(link)
        if not dt:
            text = " ".join(node.get_text(" ").split())
            dt = parse_date_from_text(text)
        if not dt or not within(dt): continue

        seen_local.add(link)
        title = a.get_text(strip=True) or "(no title)"
        rows.append({"source":"archive","feed":f"{label} (archive)","title":title,"link":link,
                     "published_utc":dt.isoformat(),"categories":label,"summary":""})
    time.sleep(REQUEST_SLEEP_SEC)
    return rows

def fetch_archives(session):
    log("[STEP] fetch archives (press/speeches/testimony) ...")
    rows = []
    start_y = to_utc_day(START_DATE).year; end_y = to_utc_day(END_DATE).year
    years = range(start_y - 1, end_y + 2)
    for y in years:
        for label, tpl in ARCHIVE_BASES.items():
            rows.extend(fetch_one_archive_list(session, tpl.format(y=y), label))
    log(f"[OK] 归档命中条目：{len(rows)}")
    rows.sort(key=lambda x: x["published_utc"], reverse=True); return rows

# ---------------- FOMC pages ----------------
FOMC_TEXT_HINTS = (
    "Statement","Postmeeting Statement","Policy Statement","Monetary Policy Statement",
    "Implementation Note","Minutes","Summary of Economic Projections","SEP",
    "Press Conference","Transcript","Addendum"
)
def looks_like_fomc_text(s: str) -> bool:
    s = s.strip()
    return any(h.lower() in s.lower() for h in FOMC_TEXT_HINTS)

def fetch_fomc_pages(session):
    log("[STEP] fetch FOMC pages ...")
    rows, seen_local = [], set()
    for page in FOMC_PAGES:
        try:
            log(f"  [FOMC] GET {page}")
            r = session.get(page, timeout=REQUEST_TIMEOUT)
            log(f"    -> status {r.status_code}")
            if r.status_code >= 400: continue
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            log(f"    ! FOMC page error: {e}"); continue

        # 扫描所有指向 /monetarypolicy/ 的链接
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href: continue
            link = href if href.startswith("http") else urljoin(page, href)
            if "/monetarypolicy/" not in link: continue
            text = a.get_text(" ", strip=True)
            if not text: continue

            # FOMC 关键词过滤（标题/链接二者满足其一即可）
            if (looks_like_fomc_text(text) or
                any(k in link.lower() for k in ["minutes","statement","implementation","sep","press","postmeeting"])):

                dt = parse_date_from_url(link) or pick_date_for_anchor(a)
                if not dt or not within(dt): continue
                if link in seen_local: continue
                seen_local.add(link)

                title = text or "(no title)"
                rows.append({
                    "source": "fomc",
                    "feed":  "FOMC (monetarypolicy pages)",
                    "title": title,
                    "link": link,
                    "published_utc": dt.isoformat(),
                    "categories": "fomc",
                    "summary": ""
                })
        time.sleep(REQUEST_SLEEP_SEC)
    log(f"[OK] FOMC 命中条目：{len(rows)}")
    rows.sort(key=lambda x: x["published_utc"], reverse=True)
    return rows

# ---------------- Output ----------------
def save_outputs(rows, csv_path: Path, json_path: Path):
    log("[STEP] save outputs ...")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source","feed","title","link","published_utc","categories","summary"])
        w.writeheader(); w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    log("[OK] files written")

def main():
    banner()
    session = build_session()
    feeds = discover_feed_urls(session)
    rows_rss  = fetch_from_rss(feeds)
    if not rows_rss:
        log("\n[提示] RSS 未命中该历史区间，转用年度归档抓取三大类（Press/Speeches/Testimony）...")
    rows_arch = fetch_archives(session)
    rows_fomc = fetch_fomc_pages(session)

    all_rows, seen = [], set()
    for r in rows_rss + rows_arch + rows_fomc:
        if r["link"] in seen: continue
        seen.add(r["link"]); all_rows.append(r)

    all_rows.sort(key=lambda x: x["published_utc"], reverse=True)
    save_outputs(all_rows, OUTPUT_CSV, OUTPUT_JSON)
    log(f"\n✅ 完成，共收集 {len(all_rows)} 条新闻（区间 {START_DATE} ~ {END_DATE}）。")
    log(f"📄 CSV:  {OUTPUT_CSV}")
    log(f"📄 JSON: {OUTPUT_JSON}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("\n[ERROR] Script crashed with exception:")
        traceback.print_exc()
        sys.exit(1)
