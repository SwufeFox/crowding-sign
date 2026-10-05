"""Download daily OHLCV + taker-flow data for four markets.

Sources (all public, no token required):
  - A-shares: Sina Finance (universe via Market_Center.getHQNodeData, klines via
    CN_MarketDataService.getKLineData). Known cap: 1023 bars per request.
  - US equities: Yahoo Finance v8/finance/chart (S&P 500 + S&P 600 current
    constituents from Wikipedia snapshots).
  - Crypto: Binance public data API (klines carry taker_buy_quote natively).

Usage:
    python src/download_ashare.py   [--outdir ak_kline] [--bars 1023]
    python src/download_us.py       [--outdir-root us_kline]
    python src/download_crypto.py   [--outdir crypto_kline] [--min-vol 5000000]
"""
