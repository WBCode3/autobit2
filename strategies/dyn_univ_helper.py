
import glob, os
import numpy as np
import pandas as pd

_UNIV_CACHE = {}

def get_univ_flags(whitelist, mode: str, top_vol: int, top_k: int):
    key = (tuple(sorted(whitelist)), mode, top_vol, top_k)
    if key in _UNIV_CACHE:
        return _UNIV_CACHE[key]
    qvols, ers, rets = {}, {}, {}
    for pair in whitelist:
        pf = pair.replace("/", "_").replace(":", "_")
        path = f"user_data/data/binance/futures/{pf}-4h-futures.feather"
        if not os.path.exists(path):
            continue
        df = pd.read_feather(path, columns=["date", "close", "volume"]).sort_values("date")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
        # Strictly causal: shift(1) so bar t only uses t-180 .. t-1
        qv = (df["close"] * df["volume"]).rolling(180, min_periods=120).sum().shift(1)
        net = (df["close"] - df["close"].shift(180)).abs()
        tot = df["close"].diff().abs().rolling(180, min_periods=120).sum()
        er = (net / tot.replace(0, np.nan)).shift(1)
        ret = (df["close"] / df["close"].shift(180) - 1.0).shift(1)
        qvols[pair] = qv
        ers[pair] = er
        rets[pair] = ret

    qv_df = pd.DataFrame(qvols)
    er_df = pd.DataFrame(ers)
    rt_df = pd.DataFrame(rets)

    # Liquid gate: top_vol by trailing 30d quote volume per timestamp
    vol_rank = qv_df.rank(axis=1, ascending=False, method="min")
    liq_mask = vol_rank <= top_vol

    if mode == "vol":
        ok_l = liq_mask
        ok_s = liq_mask
    elif mode == "er":
        er_liq = er_df.where(liq_mask)
        er_rank = er_liq.rank(axis=1, ascending=False, method="min")
        ok_l = er_rank <= top_k
        ok_s = ok_l
    elif mode == "absmom":
        am_liq = rt_df.abs().where(liq_mask)
        am_rank = am_liq.rank(axis=1, ascending=False, method="min")
        ok_l = am_rank <= top_k
        ok_s = ok_l
    elif mode == "dirmom":
        rt_liq = rt_df.where(liq_mask)
        r_hi = rt_liq.rank(axis=1, ascending=False, method="min")
        r_lo = rt_liq.rank(axis=1, ascending=True, method="min")
        ok_l = r_hi <= top_k
        ok_s = r_lo <= top_k
    else:
        ok_l = pd.DataFrame(True, index=qv_df.index, columns=qv_df.columns)
        ok_s = ok_l

    out = {"long": ok_l.fillna(False), "short": ok_s.fillna(False)}
    _UNIV_CACHE[key] = out
    return out

def get_univ_hybrid(whitelist, core_vol: int, top_vol: int, top_k: int, lookback: int = 180):
    key = ("hybrid", tuple(sorted(whitelist)), core_vol, top_vol, top_k, lookback)
    if key in _UNIV_CACHE:
        return _UNIV_CACHE[key]
    qvols, ers = {}, {}
    for pair in whitelist:
        pf = pair.replace("/", "_").replace(":", "_")
        path = f"user_data/data/binance/futures/{pf}-4h-futures.feather"
        if not os.path.exists(path):
            continue
        df = pd.read_feather(path, columns=["date", "close", "volume"]).sort_values("date")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
        mp = max(60, lookback // 2)
        qv = (df["close"] * df["volume"]).rolling(lookback, min_periods=mp).sum().shift(1)
        net = (df["close"] - df["close"].shift(lookback)).abs()
        tot = df["close"].diff().abs().rolling(lookback, min_periods=mp).sum()
        er = (net / tot.replace(0, np.nan)).shift(1)
        qvols[pair] = qv
        ers[pair] = er
    qv_df = pd.DataFrame(qvols)
    er_df = pd.DataFrame(ers)
    vol_rank = qv_df.rank(axis=1, ascending=False, method="min")
    core_mask = vol_rank <= core_vol
    liq_mask = vol_rank <= top_vol
    er_liq = er_df.where(liq_mask)
    er_rank = er_liq.rank(axis=1, ascending=False, method="min")
    ok = (core_mask | (er_rank <= top_k)).fillna(False)
    out = {"long": ok, "short": ok}
    _UNIV_CACHE[key] = out
    return out
