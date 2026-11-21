from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

class P3_1h_Breakout_LowCorr(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = "1h"
    minimal_roi = {'0': 0.36, '720': 0.18, '1440': 0.08}
    stoploss = -0.12
    process_only_new_candles = True
    startup_candle_count = 240

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return min(4.0, max_leverage)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        stake = proposed_stake * 2.0
        if min_stake is not None: stake = max(stake, min_stake)
        return min(stake, max_stake)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema24"] = ta.EMA(dataframe, timeperiod=24)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"].replace(0, np.nan)
        atr_ma = dataframe["atr_pct"].rolling(120).median()
        dataframe["atr_ratio"] = (dataframe["atr_pct"] / atr_ma.replace(0, np.nan)).fillna(1.0)
        dataframe["vol_sma"] = ta.SMA(dataframe["volume"], timeperiod=24)
        dataframe["dc_hi"] = dataframe["high"].shift(1).rolling(24).max()
        dataframe["dc_lo"] = dataframe["low"].shift(1).rolling(24).min()
        btc_1d = pd.read_feather("user_data/data/binance/futures/BTC_USDT_USDT-1d-futures.feather")
        btc_1d["btc_ema50"] = ta.EMA(btc_1d, timeperiod=50)
        btc_sub = btc_1d[["date", "btc_ema50", "close"]].rename(columns={"close": "btc_close"}).copy()
        for c in ["btc_ema50", "btc_close"]: btc_sub[c] = btc_sub[c].shift(1)
        btc_sub["date"] = btc_sub["date"].astype(dataframe["date"].dtype)
        dataframe = pd.merge_asof(dataframe.sort_values("date"), btc_sub.sort_values("date"), on="date", direction="backward")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        l = (dataframe["atr_ratio"] > 1.15) & (dataframe["atr_ratio"].shift(1) <= 1.15) & (dataframe["close"] > dataframe["dc_hi"]) & (dataframe["btc_close"] > dataframe["btc_ema50"]) & (dataframe["volume"] > dataframe["vol_sma"] * 1.5) & (dataframe["adx"] > 20)
        s = (dataframe["atr_ratio"] > 1.15) & (dataframe["atr_ratio"].shift(1) <= 1.15) & (dataframe["close"] < dataframe["dc_lo"]) & (dataframe["btc_close"] < dataframe["btc_ema50"]) & (dataframe["volume"] > dataframe["vol_sma"] * 1.5) & (dataframe["adx"] > 20)
        dataframe.loc[l & (dataframe["volume"] > 0), ["enter_long", "enter_tag"]] = (1, "1h_brk")
        dataframe.loc[s & (dataframe["volume"] > 0), ["enter_short", "enter_tag"]] = (1, "1h_brk")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["close"] < dataframe["ema24"], "exit_long"] = 1
        dataframe.loc[dataframe["close"] > dataframe["ema24"], "exit_short"] = 1
        return dataframe
