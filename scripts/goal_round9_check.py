import json, os, subprocess, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

scratch = r'C:\Users\vnddn\.gemini\antigravity-cli\brain\5d8741a9-b64b-4042-a866-eaeb7c01b10d\scratch'
g8_dir = os.path.join(scratch, 'goal8_strats')
exp8_dir = os.path.join(scratch, 'goal8_exports')
p_t10 = os.path.join(g8_dir, 'cfg_t10_714.json')
p_t10_100k = os.path.join(g8_dir, 'cfg_t10_100k.json')

with open(os.path.join(g8_dir, 'R8_Dyn90_SL18_MR2.py'), 'r', encoding='utf-8') as f:
    src = f.read()

# 1. R9_Dyn90_SL18_StrHi_MR2: combine SL=-0.18 + MR2 + StrHi (0.62/0.36/0.22)
s1 = src.replace('class R8_Dyn90_SL18_MR2(IStrategy):', 'class R9_Dyn90_SL18_StrHi_MR2(IStrategy):')
s1 = s1.replace('if losses >= 3: mult *= 0.18', 'if losses >= 3: mult *= 0.22')
s1 = s1.replace('elif losses == 2: mult *= 0.30', 'elif losses == 2: mult *= 0.36')
s1 = s1.replace('elif losses == 1: mult *= 0.55', 'elif losses == 1: mult *= 0.62')
with open(os.path.join(g8_dir, 'R9_Dyn90_SL18_StrHi_MR2.py'), 'w', encoding='utf-8') as f:
    f.write(s1)

# 2. R9_Dyn90_SL18_MR3: slightly looser idle-only MR
s2 = src.replace('class R8_Dyn90_SL18_MR2(IStrategy):', 'class R9_Dyn90_SL18_MR3(IStrategy):')
s2 = s2.replace('(dataframe["z_6"] < -2.1) & (dataframe["rsi"] < 28)', '(dataframe["z_6"] < -1.8) & (dataframe["rsi"] < 32)')
s2 = s2.replace('(dataframe["z_6"] > 2.2) & (dataframe["rsi"] > 74)', '(dataframe["z_6"] > 1.9) & (dataframe["rsi"] > 70)')
s2 = s2.replace('dataframe["vol_sma"] * 1.6', 'dataframe["vol_sma"] * 1.3')
with open(os.path.join(g8_dir, 'R9_Dyn90_SL18_MR3.py'), 'w', encoding='utf-8') as f:
    f.write(s2)

runs = [
    ('R9_Dyn90_SL18_StrHi_MR2__W1', 'R9_Dyn90_SL18_StrHi_MR2', p_t10, '20211113-20241114', '0.00035'),
    ('R9_Dyn90_SL18_StrHi_MR2__W2', 'R9_Dyn90_SL18_StrHi_MR2', p_t10, '20221114-20251114', '0.00035'),
    ('R9_Dyn90_SL18_StrHi_MR2__W3', 'R9_Dyn90_SL18_StrHi_MR2', p_t10, '20231114-20261114', '0.00035'),
    ('R9_Dyn90_SL18_StrHi_MR2__5Y', 'R9_Dyn90_SL18_StrHi_MR2', p_t10, '20211113-20261114', '0.00035'),
    ('R9_Dyn90_SL18_StrHi_MR2__26', 'R9_Dyn90_SL18_StrHi_MR2', p_t10, '20260101-20261114', '0.00035'),
    ('R9_Dyn90_SL18_StrHi_MR2__5Y_010', 'R9_Dyn90_SL18_StrHi_MR2', p_t10, '20211113-20261114', '0.001'),
    ('R9_Dyn90_SL18_MR3__W1', 'R9_Dyn90_SL18_MR3', p_t10, '20211113-20241114', '0.00035'),
    ('R9_Dyn90_SL18_MR3__W2', 'R9_Dyn90_SL18_MR3', p_t10, '20221114-20251114', '0.00035'),
    ('R9_Dyn90_SL18_MR3__W3', 'R9_Dyn90_SL18_MR3', p_t10, '20231114-20261114', '0.00035'),
    ('R9_Dyn90_SL18_MR3__5Y', 'R9_Dyn90_SL18_MR3', p_t10, '20211113-20261114', '0.00035'),
]

def run_one(item):
    label, strat, cp, tr, fee = item
    od = os.path.join(exp8_dir, label)
    os.makedirs(od, exist_ok=True)
    cmd = [
        r'.\.venv\Scripts\freqtrade.exe', 'backtesting',
        '--config', cp, '--strategy', strat, '--strategy-path', g8_dir,
        '--timerange', tr, '--timeframe-detail', '5m', '--fee', fee,
        '--export', 'trades', '--export-directory', od, '--cache', 'none'
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    zips = sorted([os.path.join(od, x) for x in os.listdir(od) if x.endswith('.zip')])
    if not zips:
        return (label, 0, 0, 0, 0, 0)
    with zipfile.ZipFile(zips[-1]) as z:
        for nm in z.namelist():
            if nm.endswith('.json') and 'meta' not in nm and 'config' not in nm:
                d = json.loads(z.read(nm))
                s = list(d['strategy'].values())[0]
                sb, fb = s['starting_balance'], s['final_balance']
                pf = s.get('profit_factor', 0)
                trs = sorted(s['trades'], key=lambda x: x['close_date'])
                eq, peak, true_dd = sb, sb, 0.0
                for t in trs:
                    eq += t['profit_abs']
                    if eq > peak: peak = eq
                    dd_now = (peak - eq) / peak * 100
                    if dd_now > true_dd: true_dd = dd_now
                return (label, len(trs), round(fb/sb, 2), round(fb*1400/1e4, 1), round(true_dd, 2), round(pf, 2))
    return (label, 0, 0, 0, 0, 0)

with ThreadPoolExecutor(max_workers=6) as ex:
    res = list(ex.map(run_one, runs))

for r in res:
    print(f'{r[0]:32s} | {r[1]:3d} tr | {r[2]:8.2f}x ({r[3]:>9,.1f}만) | true_dd {r[4]:5.2f}% | pf {r[5]:.2f}')

# Model A check for top R9 variants
vol_cache = {}
def get_pair_4h(pair):
    if pair not in vol_cache:
        pf = pair.replace('/', '_').replace(':', '_')
        df = pd.read_feather(f'user_data/data/binance/futures/{pf}-4h-futures.feather')
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values('date').set_index('date')
        df['dvol'] = df['close'] * df['volume']
        df['hl_vol'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        vol_cache[pair] = df[['dvol', 'hl_vol']]
    return vol_cache[pair]

def slip_a(label, N=1):
    od = os.path.join(exp8_dir, label)
    zp = sorted([os.path.join(od, x) for x in os.listdir(od) if x.endswith('.zip')])[-1]
    with zipfile.ZipFile(zp) as z:
        for nm in z.namelist():
            if nm.endswith('.json') and 'meta' not in nm and 'config' not in nm:
                s = list(json.loads(z.read(nm))['strategy'].values())[0]; break
    sb, fb, trs = s['starting_balance'], s['final_balance'], s['trades']
    rows, ev = [], []
    for i, t in enumerate(trs):
        d4 = get_pair_4h(t['pair'])
        odt, cdt = pd.to_datetime(t['open_date'], utc=True), pd.to_datetime(t['close_date'], utc=True)
        io, ic = d4.index.asof(odt), d4.index.asof(cdt)
        rows.append(dict(odt=odt, cdt=cdt, st=float(t['stake_amount']), lev=float(t.get('leverage', 1.0)),
            ov=float(d4.loc[io,'dvol']) if pd.notna(io) else 1e9, cv=float(d4.loc[ic,'dvol']) if pd.notna(ic) else 1e9,
            os=float(d4.loc[io,'hl_vol']) if pd.notna(io) else 0.02, cs=float(d4.loc[ic,'hl_vol']) if pd.notna(ic) else 0.02,
            pr=float(t['profit_ratio']), pa=float(t['profit_abs'])))
        ev.append((odt, 1, i)); ev.append((cdt, 0, i))
    ev.sort(key=lambda x: (x[0], x[1]))
    bal_bt, sf = sb, {}
    for _, op, i in ev:
        if op == 1: sf[i] = min(1.0, rows[i]['st'] / max(bal_bt, 1e-9))
        else: bal_bt += rows[i]['pa']
    for mname, ki, bi, ko, bo in [('ModA', 0.15, 0.0002, 0.40, 0.0004), ('ModB', 0.50, 0.0005, 0.80, 0.0007)]:
        bal, ost, eq, trace = sb, {}, [sb], []
        for _, op, i in ev:
            r = rows[i]
            if op == 1: ost[i] = min(bal * sf[i], r['st'])
            else:
                st = ost.pop(i, 0.0); mn = st * r['lev'] * N
                sr = (bi + ki * r['os'] * np.sqrt(mn / max(r['ov'], 1))) + (bo + ko * r['cs'] * np.sqrt(mn / max(r['cv'], 1)))
                bal += st * max(-1.0, r['pr'] - r['lev'] * sr)
                eq.append(bal); trace.append((r['cdt'], bal))
        ea = np.array(eq); pk = np.maximum.accumulate(ea); dd = float(((pk - ea) / pk).max() * 100)
        if mname == 'ModA':
            print(f'[Slip {label:32s} N={N}] BT={fb/sb:7.2f}x | ModA={bal/sb:7.2f}x ({bal*N*1400/1e4:9,.1f}만, dd={dd:4.1f}%)', end='')
            if '__5Y' in label and '100k' not in label:
                ts = pd.Series([v for _, v in trace], index=pd.DatetimeIndex([d for d, _ in trace]))
                yrs = []
                for yi, dt_s in enumerate(['2022-11-14', '2023-11-14', '2024-11-14', '2025-11-14', '2026-11-15'], 1):
                    sub = ts[ts.index <= pd.Timestamp(dt_s, tz='UTC')]
                    val = float(sub.iloc[-1]) if len(sub) else sb
                    yrs.append(f'Y{yi}:{val*1400/1e4:,.0f}만')
                print(' | ' + ' '.join(yrs), end='')
        else:
            print(f' | ModB={bal/sb:6.2f}x')

for l, n in [
    ('R9_Dyn90_SL18_StrHi_MR2__W1', 1), ('R9_Dyn90_SL18_StrHi_MR2__W2', 1),
    ('R9_Dyn90_SL18_StrHi_MR2__W3', 1), ('R9_Dyn90_SL18_StrHi_MR2__5Y', 1),
    ('R9_Dyn90_SL18_MR3__5Y', 1),
    ('R8_Dyn90_SL18_MR2__100k_5Y', 4), ('R8_Dyn90_SL19_StrHi__100k_5Y', 4),
]:
    slip_a(l, n)
