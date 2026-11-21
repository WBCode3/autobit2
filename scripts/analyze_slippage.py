import json, os, zipfile
import numpy as np
import pandas as pd

scratch = r'C:\Users\vnddn\.gemini\antigravity-cli\brain\5d8741a9-b64b-4042-a866-eaeb7c01b10d\scratch'

# Load 4h volume + atr_pct cache for all pairs
vol_cache = {}
def get_pair_4h(pair):
    if pair in vol_cache:
        return vol_cache[pair]
    pf = pair.replace('/', '_').replace(':', '_')
    path = f'user_data/data/binance/futures/{pf}-4h-futures.feather'
    df = pd.read_feather(path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').set_index('date')
    df['dvol'] = df['close'] * df['volume']
    # 4h high-low range volatility proxy
    df['hl_vol'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    df['dvol_ma6'] = df['dvol'].rolling(6, min_periods=1).median()
    vol_cache[pair] = df[['close', 'dvol', 'dvol_ma6', 'hl_vol']]
    return vol_cache[pair]

def analyze_export(exp_subdir, label, N=1):
    dpath = os.path.join(scratch, exp_subdir)
    zips = sorted([os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('.zip')])
    with zipfile.ZipFile(zips[-1]) as z:
        for nm in z.namelist():
            if nm.endswith('.json') and 'meta' not in nm and 'config' not in nm:
                data = json.loads(z.read(nm))
                break
    s = list(data['strategy'].values())[0]
    sb = s['starting_balance']
    fb = s['final_balance']
    trades = s['trades']

    rows = []
    for t in trades:
        pair = t['pair']
        df4 = get_pair_4h(pair)
        od = pd.to_datetime(t['open_date'], utc=True)
        cd = pd.to_datetime(t['close_date'], utc=True)
        # lookup nearest prior/exact 4h bar
        idx_o = df4.index.asof(od)
        idx_c = df4.index.asof(cd)
        ov = float(df4.loc[idx_o, 'dvol']) if pd.notna(idx_o) else 1e9
        cv = float(df4.loc[idx_c, 'dvol']) if pd.notna(idx_c) else 1e9
        osig = float(df4.loc[idx_o, 'hl_vol']) if pd.notna(idx_o) else 0.02
        csig = float(df4.loc[idx_c, 'hl_vol']) if pd.notna(idx_c) else 0.02

        lev = float(t.get('leverage', 1.0))
        stake = float(t['stake_amount'])
        notional = stake * lev  # per-account notional
        # If split across N accounts hitting the market simultaneously, total market notional = notional * N
        mkt_notional = notional * N
        pov_in = mkt_notional / max(ov, 1.0)
        pov_out = mkt_notional / max(cv, 1.0)
        rows.append({
            'pair': pair,
            'open_date': od,
            'close_date': cd,
            'stake': stake,
            'lev': lev,
            'notional': notional,
            'mkt_notional': mkt_notional,
            'ov': ov,
            'cv': cv,
            'pov_in': pov_in,
            'pov_out': pov_out,
            'osig': osig,
            'csig': csig,
            'profit_ratio': float(t['profit_ratio']), # return on stake (already includes lev & 0.035% fee)
            'profit_abs': float(t['profit_abs']),
            'exit_reason': t.get('exit_reason', ''),
        })
    df = pd.DataFrame(rows)
    print(f'\n=== {label} (N={N}, trades={len(df)}, backtest final=${fb*N:,.0f}) ===')
    print(f'Max per-acct notional: ${df["notional"].max():,.0f} | Max total-mkt notional: ${df["mkt_notional"].max():,.0f}')
    print(f'POV(entry 4h vol) median={df["pov_in"].median()*100:.4f}%, p90={df["pov_in"].quantile(0.9)*100:.3f}%, max={df["pov_in"].max()*100:.2f}%')
    print(f'POV(exit  4h vol) median={df["pov_out"].median()*100:.4f}%, p90={df["pov_out"].quantile(0.9)*100:.3f}%, max={df["pov_out"].max()*100:.2f}%')
    print('Trades with POV > 1% (total mkt notional / 4h vol):',
          f'entry={(df["pov_in"]>0.01).sum()}, exit={(df["pov_out"]>0.01).sum()}')
    print('Trades with POV > 5%:',
          f'entry={(df["pov_in"]>0.05).sum()}, exit={(df["pov_out"]>0.05).sum()}')
    top_pov = df.sort_values('pov_out', ascending=False).head(5)
    for _, r in top_pov.iterrows():
        print(f'  TopPOV: {str(r["close_date"])[:10]} {r["pair"]:14s} mkt_notional=${r["mkt_notional"]/1e6:6.2f}M / 4h_vol=${r["cv"]/1e6:6.2f}M (POV={r["pov_out"]*100:5.2f}%)')

    # Replay equity path with realistic slippage models:
    # Note: In backtest, profit_abs = stake * profit_ratio (profit_ratio is leveraged ROE).
    # We approximate proportional path scaling: track per-account equity.
    # For each trade i, stake_frac = stake / equity_before_bt.
    # Let's reconstruct backtest equity at each trade close accurately from balance trace!
    eq_bt = sb
    eq_map = []
    # Sort by open_date for replay approximation using exact profit_abs / eq_bt fraction:
    # Actually better: use exact realized balance step at close_date for closed-equity curve,
    # adjusting profit_ratio by extra slippage cost on notional = lev * extra_slip_rt!
    # Wait: extra_slip_rt (round-trip % of notional) reduces profit_ratio by lev * extra_slip_rt.
    # Two-side slippage models (entry is maker limit so lower impact; exit is taker/market so full impact):
    # Model A (Realistic Maker-in / Market-out):
    #   extra_entry = 0.0002 + 0.15 * osig * sqrt(pov_in)  (or missed/reprice cost)
    #   extra_exit  = 0.0004 + 0.40 * csig * sqrt(pov_out)  (crossing spread + impact)
    # Model B (Conservative / Harsh Taker-both + heavy impact):
    #   extra_entry = 0.0005 + 0.50 * osig * sqrt(pov_in)
    #   extra_exit  = 0.0007 + 0.80 * csig * sqrt(pov_out)
    # Model C (Extreme Whale Cap: extra impact + hard cap POV at 3% of 4h vol):
    for mname, k_in, base_in, k_out, base_out, pov_cap in [
        ('0_Backtest_Check', 0.0, 0.0, 0.0, 0.0, None),
        ('A_Realistic_MakerIn', 0.15, 0.0002, 0.40, 0.0004, None),
        ('B_Harsh_Market', 0.50, 0.0005, 0.80, 0.0007, None),
        ('C_Harsh_POVcap3pct', 0.50, 0.0005, 0.80, 0.0007, 0.03),
    ]:
        # Recompute sequential/overlapping equity cleanly by matching trades' close_date order
        # First pass on 0_Backtest_Check tells us exact stake_fraction relative to bt closed balance
        pass

    # Better: compute exact stake_pct_of_bt_bal at each trade open by replaying bt balance timeline
    events = []
    for i, r in df.iterrows():
        events.append((r['open_date'], 1, i))
        events.append((r['close_date'], 0, i))
    events.sort(key=lambda x: (x[0], x[1]))

    bal_bt = sb
    open_stakes_bt = {}
    stake_fracs = {}
    for dt, is_open, i in events:
        if is_open == 1:
            stake_fracs[i] = min(1.0, df.loc[i, 'stake'] / max(bal_bt, 1e-9))
            open_stakes_bt[i] = df.loc[i, 'stake']
        else:
            open_stakes_bt.pop(i, 0.0)
            bal_bt += df.loc[i, 'profit_abs']

    for mname, k_in, base_in, k_out, base_out, pov_cap in [
        ('0_Replay_Check', 0.0, 0.0, 0.0, 0.0, None),
        ('A_Realistic_MakerIn', 0.15, 0.0002, 0.40, 0.0004, None),
        ('B_Harsh_Impact', 0.50, 0.0005, 0.80, 0.0007, None),
        ('C_Harsh_POVcap3pct', 0.50, 0.0005, 0.80, 0.0007, 0.03),
    ]:
        bal = sb
        open_st = {}
        eq_curve = [sb]
        slips_bps = []
        capped_cnt = 0
        for dt, is_open, i in events:
            r = df.loc[i]
            if is_open == 1:
                st_new = bal * stake_fracs[i]
                st_new = min(st_new, r['stake'])
                if pov_cap is not None:
                    max_mkt_notional = min(r['ov'], r['cv']) * pov_cap
                    max_acct_stake = max_mkt_notional / (N * max(r['lev'], 1.0))
                    if st_new > max_acct_stake:
                        st_new = max_acct_stake
                        capped_cnt += 1
                open_st[i] = st_new
            else:
                st_new = open_st.pop(i, 0.0)
                if mname == '0_Replay_Check':
                    slip_rt = 0.0
                else:
                    # dynamic POV using actual mkt_notional under this model!
                    mkt_not = st_new * r['lev'] * N
                    p_in = mkt_not / max(r['ov'], 1.0)
                    p_out = mkt_not / max(r['cv'], 1.0)
                    s_in = base_in + k_in * r['osig'] * np.sqrt(p_in)
                    s_out = base_out + k_out * r['csig'] * np.sqrt(p_out)
                    slip_rt = s_in + s_out
                    slips_bps.append(slip_rt * 10000)
                adj_roe = max(-1.0, r['profit_ratio'] - r['lev'] * slip_rt)
                bal += st_new * adj_roe
                eq_curve.append(bal)
        eq_arr = np.array(eq_curve)
        peak = np.maximum.accumulate(eq_arr)
        dd = ((peak - eq_arr) / peak).max() * 100
        tot_final = bal * N
        tot_start = sb * N
        avg_bps = np.mean(slips_bps) if slips_bps else 0.0
        p90_bps = np.percentile(slips_bps, 90) if slips_bps else 0.0
        print(f'  {mname:20s}: final=${tot_final:>13,.0f} ({tot_final/tot_start:6.2f}x) | dd~{dd:5.2f}% | avg_slip={avg_bps:5.1f}bps p90={p90_bps:5.1f}bps | pov_caps={capped_cnt}')

for sub, lbl, N in [
    ('final_krw_exports/M1_5y_100k_N1', 'M1_5y_100k_N1 ($100k->5y)', 1),
    ('final_krw_exports/M1_5y_100k_N4', 'M1_5y_100k_N4 ($100k->5y Split4)', 4),
    ('final_krw_exports/M2_5y_100k_N1', 'M2_5y_100k_N1 ($100k->5y)', 1),
    ('tune3_exports/M1_m44_2m', 'M1_1y_2M_N1 ($2M->1y)', 1),
    ('tune3_exports/M1_m44_5m', 'M1_1y_5M_N1 ($5M->1y)', 1),
    ('tune3_exports/M1_m44_N4_5m', 'M1_1y_5M_N4 ($5M->1y Split4)', 4),
    ('tune3b_exports/M2_12_4_m40_5m', 'M2_1y_5M_N1 ($5M->1y)', 1),
]:
    analyze_export(sub, lbl, N)
