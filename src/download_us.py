"""US equities daily klines (Yahoo Finance) for S&P 500 + S&P 600 constituents.

Constituent lists are scraped from Wikipedia snapshots at run time (the same
lists used for the paper: survivorship caveat applies -- see README).
"""
import concurrent.futures as cf
import json
import os
import re
import time
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
YCHART = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
          "?range=5y&interval=1d")


def http_json(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wiki_tickers(url: str) -> list[str]:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    i = html.find("constituents")
    seg = html[i:i + 2_000_000]
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", seg, re.S)
    ticks = []
    for r in rows:
        tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)
        if not tds:
            continue
        sym = re.sub(r"<[^>]+>", "", tds[0]).strip().replace("&amp;", "&")
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", sym):
            ticks.append(sym)
    seen, out = set(), []
    for t in ticks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def fetch_daily(sym: str):
    d = http_json(YCHART.format(sym=sym.replace(".", "-")))
    res = d.get("chart", {}).get("result")
    if not res:
        return None
    q = res[0]["indicators"]["quote"][0]
    ts = res[0]["timestamp"]
    return [{"time": time.strftime("%Y-%m-%d", time.gmtime(x)),
             "open": o, "close": c, "volume": v}
            for x, o, c, v in zip(ts, q["open"], q["close"], q["volume"])
            if c is not None and v is not None]


def download(root: str = "us_kline") -> None:
    for name, wiki in WIKI.items():
        outdir = os.path.join(root, name)
        os.makedirs(outdir, exist_ok=True)
        ticks = wiki_tickers(wiki)
        print(f"{name}: {len(ticks)} tickers", flush=True)

        def work(t: str):
            path = os.path.join(outdir, f"{t}.json")
            if os.path.exists(path) and os.path.getsize(path) > 2000:
                return t, "cached"
            try:
                rows = fetch_daily(t)
                if not rows or len(rows) < 400:
                    return t, "short"
                json.dump(rows, open(path, "w"))
                return t, "ok"
            except Exception as e:  # noqa: BLE001
                return t, f"ERR {type(e).__name__}"

        ok = fail = 0
        errs = []
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            for t, s in ex.map(work, ticks):
                if s in ("ok", "cached"):
                    ok += 1
                else:
                    fail += 1
                    errs.append((t, s))
        print(f"{name}: ok={ok} fail={fail}; sample errors: {errs[:5]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir-root", default="data")
    a = ap.parse_args()
    download(a.outdir_root)
