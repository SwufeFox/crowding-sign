"""A-share universe + daily klines from Sina Finance (public, no token).

Threading is modest (6 workers) and pages are fetched serially with small
sleeps: both endpoints rate-limit aggressively. Eastmoney push2 is NOT used
(it returns HTTP 501 after a few hundred requests per IP).
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

import pandas as pd

SINA_LIST = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
             "Market_Center.getHQNodeData?page={page}&num={num}&sort=symbol&asc=1&node=hs_a")
SINA_KLINE = ("https://quotes.sina.cn/cn/api/json_v2.php/"
              "CN_MarketDataService.getKLineData?symbol={sym}&scale=240&ma=no&datalen={n}")
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}


def http_json(url: str, retries: int = 4, encoding: str = "gbk"):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode(encoding, errors="replace"))
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last = e
            time.sleep(1.0 + attempt)
    raise last  # type: ignore[misc]


def fetch_universe() -> pd.DataFrame:
    """All HS-A snapshot rows (~5500), includes turnoverratio / mktcap / nmc."""
    rows, page = [], 1
    while True:
        d = http_json(SINA_LIST.format(page=page, num=80))
        if not d:
            break
        rows.extend(d)
        if len(d) < 80:
            break
        page += 1
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str)
    return df


def download(outdir: str = "ak_kline", bars: int = 1023, min_bars: int = 250) -> None:
    os.makedirs(outdir, exist_ok=True)
    snap = fetch_universe()
    # main board (60/00) + ChiNext (30) + STAR (68); drop BSE (4/8/9 prefixes)
    codes = [c for c in snap["code"].tolist() if c[0] in "0356"]
    print(f"universe: {len(codes)} A-share codes (excl. BSE)", flush=True)

    def work(code: str):
        path = os.path.join(outdir, f"{code}.json")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            return code, "cached"
        sym = ("sh" if code.startswith("6") else "sz") + code
        try:
            d = http_json(SINA_KLINE.format(sym=sym, n=bars))
            rows = [{"time": r["day"], "open": float(r["open"]), "close": float(r["close"]),
                     "high": float(r["high"]), "low": float(r["low"]),
                     "volume": float(r["volume"])} for r in d]
            if len(rows) < min_bars:
                return code, "short"
            json.dump(rows, open(path, "w"))
            return code, "ok"
        except Exception as e:  # noqa: BLE001
            return code, f"ERR {type(e).__name__}"

    ok = fail = 0
    errs = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for i, (c, s) in enumerate(ex.map(work, codes)):
            if s in ("ok", "cached"):
                ok += 1
            else:
                fail += 1
                errs.append((c, s))
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(codes)} ok={ok} fail={fail}", flush=True)
    print(f"done: ok={ok} fail={fail}; files={len(os.listdir(outdir))}")
    if errs:
        print("sample errors:", errs[:10])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/ak_kline")
    ap.add_argument("--bars", type=int, default=1023)
    ap.add_argument("--min-bars", type=int, default=250)
    a = ap.parse_args()
    sys.exit(download(a.outdir, a.bars, a.min_bars) or 0)
