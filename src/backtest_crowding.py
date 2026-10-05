"""Four-market crowding backtest -- the paper's main experiment.

Protocol (identical across markets):
  factors (high = crowded): F1 20d momentum, F2 dollar-volume 20/120 ratio,
  F3 |return|>threshold day count (20d), F4 20d corr(return, dvolume).
  Crypto adds C4 taker-buy-ratio deviation (MA20 - MA120) and C5 trade-count
  ratio. Cross-sectional percentile-rank, equal-weight composite.
  Month-end (or week-end) quintile/tercile portfolios, hold next period,
  report Q5-Q1 spread series: monthly mean, t-stat, IR, positive months.

Also runs: A-share limit-up event study (magnet -> reversal lifecycle) and
the crypto C4 robustness battery (momentum-orthogonalized, weekly, level
vs deviation, per-year).

Expected headline outputs (paper, 2026-10-05 data window):
  A-shares   Q5-Q1 = -1.18%/mo (t=-1.32); limit-up weekly t=-2.72
  S&P 500    Q5-Q1 = +0.91%/mo (t=+1.91)
  S&P 600    Q5-Q1 = +0.30%/mo (t=+0.74)
  crypto C4  majors -5.25%/mo (t=-2.59); orthogonalized t=-2.07
Exact numbers drift with data window; signs and magnitudes should hold.
"""
import json
import math
import os

import numpy as np
import pandas as pd

CROWDING_FACTORS = ["F1_mom20", "F2_volspike", "F3_jump", "F4_flowcorr"]
CRYPTO_FACTORS = CROWDING_FACTORS + ["C4_takerdev", "C5_trades"]
MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "ADAUSDT",
          "DOGEUSDT", "TRXUSDT", "LTCUSDT", "LINKUSDT", "BCHUSDT", "DOTUSDT"}


# ---------- data loading ----------

def load_universe(dirname: str, min_bars: int = 400,
                  crypto: bool = False):
    """Return (close, volume) or (close, qvol, taker, trades) wide frames."""
    px, vol, tb, tr = {}, {}, {}, {}
    for f in sorted(os.listdir(dirname)):
        c = f[:-5]
        rows = json.load(open(os.path.join(dirname, f)))
        if len(rows) < min_bars:
            continue
        px[c] = pd.Series({r["time"]: r["close"] for r in rows})
        # crypto rows use quote_vol (quote-currency turnover) as the volume proxy
        vol[c] = pd.Series({r["time"]: r.get("volume", r.get("quote_vol"))
                            for r in rows})
        if crypto:
            tb[c] = pd.Series({r["time"]: r["taker_buy_quote"] for r in rows})
            tr[c] = pd.Series({r["time"]: r["trades"] for r in rows})
    close = pd.DataFrame(px).sort_index()
    close.index = pd.to_datetime(close.index)
    close = close.dropna(axis=1, how="all")
    volume = pd.DataFrame(vol).sort_index()
    volume.index = close.index
    volume = volume.reindex(close.index)[close.columns]
    if crypto:
        taker = pd.DataFrame(tb).sort_index()
        taker.index = close.index
        trades = pd.DataFrame(tr).sort_index()
        trades.index = close.index
        return close, volume, taker, trades
    return close, volume


# ---------- factors ----------

def build_factors(close: pd.DataFrame, volume: pd.DataFrame,
                  jump_thr: float = 0.05, crypto: bool = False,
                  taker: pd.DataFrame | None = None,
                  trades: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    rets = close.pct_change(fill_method=None)
    dva = close * volume
    out = {
        "F1_mom20": close.pct_change(20, fill_method=None),
        "F2_volspike": dva.rolling(20).mean() / dva.rolling(120).mean(),
        "F3_jump": (rets.abs() > jump_thr).rolling(20).sum(),
        "F4_flowcorr": rets.rolling(20).corr(
            volume.pct_change(fill_method=None).clip(-1, 1)),
    }
    if crypto and taker is not None and trades is not None:
        tbr = (taker / volume.where(volume > 0)).clip(0.01, 0.99)
        out["C4_takerdev"] = tbr.rolling(20).mean() - tbr.rolling(120).mean()
        out["C5_trades"] = (trades.rolling(20).mean()
                            / trades.rolling(120).mean())
    return out


# ---------- portfolio backtest ----------

def composite_quintiles(close, volume, factors: dict, label: str,
                        hold: str = "M", nbuckets: int = 5,
                        subset: pd.Series | None = None,
                        min_names: int = 80,
                        active_rule=lambda vol: vol.iloc[-5:].mean() > 0):
    """Equal-weight quintile portfolios on the composite crowding score."""
    month_ends = close.resample(f"{hold}E").last().index
    all_dates = close.index
    rows = []
    for i in range(len(month_ends) - 1):
        d0, d1 = month_ends[i], month_ends[i + 1]
        hs = all_dates[all_dates > d0][:1]
        if len(hs) == 0:
            continue
        hs = hs[0]
        p0 = close.loc[:hs].iloc[-1]
        p1 = close.loc[:d1].iloc[-1]
        fwd = p1 / p0 - 1
        active = active_rule(volume)
        ranks = []
        for F in factors.values():
            sig = F.loc[F.index <= d0].iloc[-1]
            ranks.append(sig[active].rank(pct=True))
        score = pd.concat(ranks, axis=1).mean(axis=1)
        cnt = (score.notna() & p0.notna() & fwd.notna()
               & active.reindex(score.index).fillna(False))
        if subset is not None:
            cnt = cnt & subset.reindex(score.index).fillna(False)
        s, r = score[cnt], fwd[cnt]
        if len(s) < min_names:
            continue
        q = pd.qcut(s.rank(method="first"), nbuckets, labels=False)
        rows.append({"date": hs,
                     "spread": r[q == nbuckets - 1].mean() - r[q == 0].mean()})
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        print(f"{label}: no valid months")
        return None
    m, sd = df["spread"].mean(), df["spread"].std()
    t = m / (sd / math.sqrt(len(df))) if sd > 0 else 0.0
    per = {"M": 12, "W": 52}[hold]
    print(f"{label}: n={len(df)} spread={m*100:+.2f}%/period "
          f"t={t:+.2f} IR={m/sd*math.sqrt(per):+.2f} "
          f"pos={int((df['spread']>0).sum())}/{len(df)}")
    return df


# ---------- A-share limit-up event study ----------

def limit_up_event_study(close: pd.DataFrame):
    """Lifecycle of limit-up closes: T+1 magnet, 20d/60d reversal."""
    prev = close.shift(1)
    ret = close / prev - 1
    lim = np.where(pd.Series([c[:2] for c in close.columns])
                   .isin(["30", "68"]).values, 0.1996, 0.0998)
    thr = prev.values * (1 + lim[None, :])
    is_limit = pd.DataFrame(close.values >= thr,
                            index=close.index, columns=close.columns)
    n_events = int(is_limit.values.sum())
    print(f"\n=== A-share limit-up event study ({n_events} events) ===")
    next_ret = ret.shift(-1)
    nv, iv = next_ret.values, is_limit.values
    mask = iv & ~np.isnan(nv)
    all_next = nv[~np.isnan(nv)]
    print(f"T+1: limit-up {nv[mask].mean()*100:+.3f}% vs all {all_next.mean()*100:+.3f}% "
          f"(neg-prob {(nv[mask]<0).mean()*100:.1f}% vs {(all_next<0).mean()*100:.1f}%)")
    # consecutive limit-up runs (column-wise run counter)
    run = np.zeros_like(iv, dtype=float)
    for i in range(iv.shape[0]):
        run[i] = iv[i] * (run[i - 1] + 1) if i > 0 else iv[i]
    for k in (2, 3):
        m = (run >= k) & ~np.isnan(nv)
        if m.sum() > 100:
            print(f">={k} consecutive: {nv[m].mean()*100:+.3f}% (n={int(m.sum())})")
    for w in (5, 20, 60):
        fwd = (close.shift(-w) / close - 1).values
        mm = iv & ~np.isnan(fwd)
        af = fwd[~np.isnan(fwd)]
        print(f"{w}d: {np.nanmean(np.where(mm, fwd, np.nan))*100:+.3f}% "
              f"vs all {np.nanmean(af)*100:+.3f}%")
    # per-year 20d relative gap (paper: five consecutive negatives)
    fwd20 = (close.shift(-20) / close - 1).values
    years = close.index.year.values
    print("20d relative gap by year:")
    for y in sorted(set(years)):
        my = ((years == y)[:, None] & iv & ~np.isnan(fwd20))
        ma = ((years == y)[:, None] & ~np.isnan(fwd20))
        a = np.nanmean(np.where(my, fwd20, np.nan)) * 100
        b = np.nanmean(np.where(ma, fwd20, np.nan)) * 100
        print(f"  {y}: {a:+.2f}% vs {b:+.2f}%  gap {a-b:+.2f}pp")


# ---------- crypto C4 robustness ----------

def crypto_c4_robustness(close, volume, taker):
    tbr = (taker / volume.where(volume > 0)).clip(0.01, 0.99)
    f4 = tbr.rolling(20).mean() - tbr.rolling(120).mean()
    f1 = close.pct_change(20, fill_method=None)
    major = pd.Series({c: c in MAJORS for c in close.columns})
    month_ends = close.resample("ME").last().index
    all_dates = close.index

    def spread(sig, subset, orthogonalize=False):
        rows = []
        for i in range(len(month_ends) - 1):
            d0, d1 = month_ends[i], month_ends[i + 1]
            hs = all_dates[all_dates > d0][:1]
            if len(hs) == 0:
                continue
            hs = hs[0]
            p0 = close.loc[:hs].iloc[-1]
            p1 = close.loc[:d1].iloc[-1]
            fwd = p1 / p0 - 1
            active = volume.loc[:hs].iloc[-5:].mean() > 1_000_000
            s4 = sig.loc[sig.index <= d0].iloc[-1]
            cnt = (s4.notna() & p0.notna() & fwd.notna()
                   & active.reindex(s4.index).fillna(False)
                   & subset.reindex(s4.index).fillna(False))
            if cnt.sum() < 10:
                continue
            y = s4[cnt]
            if orthogonalize:
                x = f1.loc[f1.index <= d0].iloc[-1][cnt]
                beta = ((x - x.mean()) * (y - y.mean())).sum() / \
                       ((x - x.mean()) ** 2).sum()
                y = y - beta * x
            r = fwd[cnt]
            q = pd.qcut(y.rank(method="first"), 3, labels=False)
            rows.append(r[q == 2].mean() - r[q == 0].mean())
        return pd.Series(rows)

    for name, sig, sub, orth in [
        ("C4 majors (deviation)", f4, major, False),
        ("C4 majors (⊥ momentum)", f4, major, True),
        ("C4 all", f4, pd.Series(True, index=close.columns), False),
    ]:
        d = spread(sig, sub, orth)
        m, sd = d.mean(), d.std()
        t = m / (sd / math.sqrt(len(d))) if sd > 0 else 0
        print(f"{name}: {m*100:+.2f}%/mo t={t:+.2f} pos={int((d>0).sum())}/{len(d)}")


# ---------- main ----------

def main():
    here = os.environ.get("CROWDING_DATA", "..")

    print("== A-shares (Sina, full market) ==", flush=True)
    ak_close, ak_vol = load_universe(os.path.join(here, "ak_kline"))
    print(f"matrix: {ak_close.shape}")
    akF = build_factors(ak_close, ak_vol)
    composite_quintiles(ak_close, ak_vol,
                        {k: akF[k] for k in CROWDING_FACTORS},
                        "A-share composite (monthly quintiles)")
    limit_up_event_study(ak_close)

    print("\n== S&P 500 ==", flush=True)
    sp_close, sp_vol = load_universe(os.path.join(here, "sp500"))
    print(f"matrix: {sp_close.shape}")
    spF = build_factors(sp_close, sp_vol)
    composite_quintiles(sp_close, sp_vol,
                        {k: spF[k] for k in CROWDING_FACTORS},
                        "S&P500 composite (monthly quintiles)")

    print("\n== S&P 600 ==", flush=True)
    sm_close, sm_vol = load_universe(os.path.join(here, "sp600"))
    print(f"matrix: {sm_close.shape}")
    smF = build_factors(sm_close, sm_vol)
    composite_quintiles(sm_close, sm_vol,
                        {k: smF[k] for k in CROWDING_FACTORS},
                        "S&P600 composite (monthly quintiles)")

    print("\n== Crypto (Binance spot) ==", flush=True)
    c_close, c_vol, c_taker, c_tr = load_universe(
        os.path.join(here, "crypto_kline"), crypto=True)
    print(f"matrix: {c_close.shape}")
    cF = build_factors(c_close, c_vol, jump_thr=0.07, crypto=True,
                       taker=c_taker, trades=c_tr)
    composite_quintiles(c_close, c_vol,
                        {k: cF[k] for k in CROWDING_FACTORS},
                        "crypto price-volume composite (terciles)",
                        nbuckets=3, min_names=10,
                        active_rule=lambda v: v.iloc[-5:].mean() > 1_000_000)
    crypto_c4_robustness(c_close, c_vol, c_taker)

    print("\nPaper headline (2026-10-05 window):")
    print("  A-shares -1.18%/mo t=-1.32 | S&P500 +0.91% t=+1.91 | "
          "S&P600 +0.30% t=+0.74 | C4 majors -5.25% t=-2.59 (⊥ t=-2.07)")


if __name__ == "__main__":
    main()
