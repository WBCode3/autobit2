import subprocess, os
from concurrent.futures import ThreadPoolExecutor
s=r'C:\Users\vnddn\.gemini\antigravity-cli\brain\5d8741a9-b64b-4042-a866-eaeb7c01b10d\scratch\goal8_strats'
runs=[(st,tr) for st in ['R9_Dyn90_SL18_StrHi_MR2','R8_Dyn85_SL18'] for tr in ['20261114-20261122','20261022-20261122','20260822-20261122']]
def go(r):
    st,tr=r
    exp=os.path.join(s,'..','goal_now_'+st+'_'+tr)
    os.makedirs(exp,exist_ok=True)
    p=subprocess.run([r'.\.venv\Scripts\freqtrade.exe','backtesting','--config',s+r'\cfg_t10_500.json','--strategy',st,'--strategy-path',s,'--timerange',tr,'--timeframe-detail','5m','--fee','0.00035','--cache','none','--export','trades','--export-directory',exp],capture_output=True,text=True)
    out=p.stdout+p.stderr
    keep=[l for l in out.splitlines() if any(k in l for k in ['TOTAL','Total profit','Absolute drawdown','Max % of account','Final balance','Starting balance','Worst Pair','Best Pair','Drawdown','No trades'])]
    return st,tr,p.returncode,'\n'.join(keep) or out[-800:]
with ThreadPoolExecutor(3) as ex:
    for st,tr,rc,o in ex.map(go,runs): print('=====',st,tr,rc);print(o)
