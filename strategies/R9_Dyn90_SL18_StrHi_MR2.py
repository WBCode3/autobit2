from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class R9_Dyn90_SL18_StrHi_MR2(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = "4h"
    minimal_roi = {'0': 0.80, '1440': 0.42, '2880': 0.20}
    stoploss = -0.18
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 240

    def custom_entry_price(self, pair: str, trade, current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last = df.iloc[-1]
            dc = float(last["dc_hi"] if side == "long" else last["dc_lo"])
            mid = 0.75 * proposed_rate + 0.25 * dc
            if side == "long" and 0.985 * proposed_rate <= mid < proposed_rate: return mid
            if side == "short" and proposed_rate < mid <= 1.015 * proposed_rate: return mid
        except Exception:
            pass
        return proposed_rate * (0.9975 if side == "long" else 1.0025)

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        if trade.enter_tag == "mr":
            if current_profit >= 0.16: return "mr_tp"
            if current_profit <= -0.12: return "mr_sl"
            if (current_time - trade.open_date_utc) >= timedelta(hours=32): return "mr_to"
            return None
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last = df.iloc[-1]
            if not trade.is_short and float(last["close"]) < float(last["ema20"]): return "exit_signal"
            if trade.is_short and float(last["close"]) > float(last["ema20"]): return "exit_signal"
        except Exception:
            pass
        return None

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 1},
            {"method": "StoplossGuard", "lookback_period_candles": 12, "trade_limit": 1, "stop_duration_candles": 4, "only_per_pair": False},
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            adx_v = float(df.iloc[-1].get("adx", 25.0))
            if adx_v >= 32: return min(8.0, max_leverage)
            elif adx_v >= 22: return min(7.0, max_leverage)
            return min(5.0, max_leverage)
        except Exception:
            return min(7.0, max_leverage)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        mult = 7.2 if entry_tag == "brk" else 4.0
        try:
            n_open = len(Trade.get_trades_proxy(is_open=True))
            if entry_tag == "mr" and n_open >= 1:
                return 0.0
            if n_open == 1: mult = 4.4
            elif n_open == 2: mult = 2.6
            elif n_open >= 3: mult = 1.6
            closed = sorted([t for t in Trade.get_trades_proxy(is_open=False) if t.close_date_utc is not None], key=lambda t: t.close_date_utc, reverse=True)
            losses = 0
            for t in closed[:5]:
                if (t.close_profit or 0.0) < 0: losses += 1
                else: break
            if losses >= 3: mult *= 0.22
            elif losses == 2: mult *= 0.3
            elif losses == 1: mult *= 0.62
        except Exception:
            pass
        stake = proposed_stake * mult
        if min_stake is not None: stake = max(stake, min_stake)
        return min(stake, max_stake)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"].replace(0, np.nan)
        atr_ma = dataframe["atr_pct"].rolling(100).median()
        dataframe["atr_ratio"] = (dataframe["atr_pct"] / atr_ma.replace(0, np.nan)).fillna(1.0)
        dataframe["vol_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)
        dataframe["dc_hi"] = dataframe["high"].shift(1).rolling(8).max()
        dataframe["dc_lo"] = dataframe["low"].shift(1).rolling(8).min()
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_up"] = bb["upperband"]
        dataframe["bb_lo"] = bb["lowerband"]
        ret6 = dataframe["close"].pct_change(6)
        sig6 = dataframe["atr_pct"] * np.sqrt(6.0)
        dataframe["z_6"] = (ret6 / sig6.replace(0, np.nan)).fillna(0.0)

        btc_1d = pd.read_feather("user_data/data/binance/futures/BTC_USDT_USDT-1d-futures.feather")
        btc_1d["btc_ema50"] = ta.EMA(btc_1d, timeperiod=50)
        btc_sub = btc_1d[["date", "btc_ema50", "close"]].rename(columns={"close": "btc_close"}).copy()
        for c in ["btc_ema50", "btc_close"]:
            btc_sub[c] = btc_sub[c].shift(1)
        btc_sub["date"] = btc_sub["date"].astype(dataframe["date"].dtype)
        dataframe = pd.merge_asof(dataframe.sort_values("date"), btc_sub.sort_values("date"), on="date", direction="backward")

        pf = metadata["pair"].replace("/", "_").replace(":", "_")
        fr = pd.read_feather(f"user_data/data/binance/futures/{pf}-1h-funding_rate.feather")
        fc = "funding_rate" if "funding_rate" in fr.columns else "open"
        fr = fr[["date", fc]].rename(columns={fc: "funding_rate"})
        fr["date"] = fr["date"].astype(dataframe["date"].dtype)
        dataframe = pd.merge_asof(dataframe.sort_values("date"), fr.sort_values("date"), on="date", direction="backward")
        dataframe["funding_rate"] = dataframe["funding_rate"].fillna(0.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        l1 = (dataframe["atr_ratio"] > 1.08) & (dataframe["atr_ratio"].shift(1) <= 1.08) & (dataframe["close"] > dataframe["dc_hi"]) & (dataframe["btc_close"] > dataframe["btc_ema50"]) & (dataframe["volume"] > dataframe["vol_sma"] * 1.1) & (dataframe["funding_rate"] < 0.0004)
        s1 = (dataframe["atr_ratio"] > 1.08) & (dataframe["atr_ratio"].shift(1) <= 1.08) & (dataframe["close"] < dataframe["dc_lo"]) & (dataframe["btc_close"] < dataframe["btc_ema50"]) & (dataframe["volume"] > dataframe["vol_sma"] * 1.1) & (dataframe["funding_rate"] > -0.0002)
        mrl = (dataframe["z_6"] < -2.1) & (dataframe["rsi"] < 28) & (dataframe["close"] < dataframe["bb_lo"]) & (dataframe["volume"] > dataframe["vol_sma"] * 1.6) & (dataframe["funding_rate"] < 0.0001) & (dataframe["close"] > dataframe["open"])
        mrs = (dataframe["z_6"] > 2.2) & (dataframe["rsi"] > 74) & (dataframe["close"] > dataframe["bb_up"]) & (dataframe["volume"] > dataframe["vol_sma"] * 1.6) & (dataframe["funding_rate"] > 0.00015) & (dataframe["close"] < dataframe["open"])
        dataframe.loc[mrl & (dataframe["volume"] > 0), ["enter_long", "enter_tag"]] = (1, "mr")
        dataframe.loc[mrs & (dataframe["volume"] > 0), ["enter_short", "enter_tag"]] = (1, "mr")
        dataframe.loc[l1 & (dataframe["volume"] > 0), ["enter_long", "enter_tag"]] = (1, "brk")
        dataframe.loc[s1 & (dataframe["volume"] > 0), ["enter_short", "enter_tag"]] = (1, "brk")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
