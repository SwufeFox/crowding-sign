"""Binance spot daily klines (public data API, no token).

Universe: USDT pairs with 24h quote volume above threshold (default $5M),
stablecoins excluded. Klines natively carry taker_buy_quote (field 10) --
the paper's crypto crowding signal is built on it.
"""
import concurrent.futures as cf
import json
import os
import time
import urllib.request

BASE = "https://data-api.binance.vision"
STABLES = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "DAIUSDT",
           "USD1USDT", "EURUSDT", "BUSDUSDT", "AEURUSDT", "USDEUSDT", "EURIUSDT"}
HEADERS = {"User-Agent": "Mozilla/5.0"}


def http_json(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def universe(min_vol: float) -> list[str]:
    t = http_json(f"{BASE}/api/v3/ticker/24hr", timeout=30)
    cands = [x for x in t
             if x["symbol"].endswith("USDT") and x["symbol"] not in STABLES
             and float(x["quoteVolume"]) > min_vol
             and x.get("status", "TRADING") != "BREAK"]
    cands.sort(key=lambda x: -float(x["quoteVolume"]))
    return [x["symbol"] for x in cands]


def fetch_daily(sym: str, start: str = "2022-01-01"):
    ms0 = int(time.mktime(time.strptime(start, "%Y-%m-%d"))) * 1000
    out, cur = [], ms0
    while True:
        d = http_json(f"{BASE}/api/v3/klines?symbol={sym}&interval=1d"
                      f"&startTime={cur}&limit=1000")
        if not d:
            break
        out.extend(d)
        if len(d) < 1000:
            break
        cur = d[-1][6] + 1
    return [{"time": time.strftime("%Y-%m-%d", time.gmtime(x[0] / 1000)),
             "open": float(x[1]), "close": float(x[4]),
             "quote_vol": float(x[7]), "trades": int(x[8]),
             "taker_buy_quote": float(x[10])} for x in out]


def download(outdir: str = "crypto_kline", min_vol: float = 5_000_000,
             min_bars: int = 400) -> None:
    os.makedirs(outdir, exist_ok=True)
    syms = universe(min_vol)
    print(f"universe: {len(syms)} USDT pairs (>{min_vol:,.0f}/day)", flush=True)

    def work(sym: str):
        path = os.path.join(outdir, f"{sym}.json")
        if os.path.exists(path) and os.path.getsize(path) > 3000:
            return sym, "cached"
        try:
            rows = fetch_daily(sym)
            if len(rows) < min_bars:
                return sym, "short"
            json.dump(rows, open(path, "w"))
            return sym, "ok"
        except Exception as e:  # noqa: BLE001
            return sym, f"ERR {type(e).__name__}"

    ok = fail = 0
    errs = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for s, st in ex.map(work, syms):
            if st in ("ok", "cached"):
                ok += 1
            else:
                fail += 1
                errs.append((s, st))
    print(f"done: ok={ok} fail={fail}; files={len(os.listdir(outdir))}")
    if errs:
        print("sample errors:", errs[:8])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/crypto_kline")
    ap.add_argument("--min-vol", type=float, default=5_000_000)
    a = ap.parse_args()
    download(a.outdir, a.min_vol)
