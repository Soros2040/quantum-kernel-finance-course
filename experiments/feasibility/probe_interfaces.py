"""Fixed public-market API probes. No authentication lookup or source fallback."""
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import time
from queue import Empty
from safe_io import OutputRoot


def worker(spec, root, queue):
    start=time.perf_counter()
    try:
        if spec['source']=='akshare':
            import akshare as ak
            table=ak.stock_zh_a_hist(symbol=spec['symbol'],period='daily',start_date=spec['start'],end_date=spec['end'],adjust=spec['adjust'])
        else:
            import yfinance as yf
            yf.set_tz_cache_location(str(OutputRoot(root).path('public_api_cache')))
            table=yf.download('SPY',start='2024-01-02',end='2024-01-30',auto_adjust=False,progress=False,threads=False,timeout=20)
        text=table.to_csv(index=True)
        output=OutputRoot(root).path(spec['id']+'.csv')
        output.write_text(text,encoding='utf-8')
        queue.put({**spec,'status':'returned' if len(table) else 'empty','rows':len(table),'columns':[str(c) for c in table.columns],
                   'sha256':hashlib.sha256(text.encode()).hexdigest(),'seconds':time.perf_counter()-start})
    except Exception as exc:
        queue.put({**spec,'status':'failed','error_type':type(exc).__name__,'message':str(exc)[:500],'seconds':time.perf_counter()-start})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--event-start',default='20240715');p.add_argument('--event-end',default='20240725')
    args=p.parse_args();out=OutputRoot(args.output)
    specs=[]
    for symbol in ('600000','600004'):
        for adjust in ('','qfq','hfq'):
            specs.append({'id':symbol+'_'+(adjust or 'raw'),'source':'akshare','symbol':symbol,'start':'20240102','end':'20240129','adjust':adjust})
    for adjust in ('','qfq','hfq'):
        specs.append({'id':'event_'+(adjust or 'raw'),'source':'akshare','symbol':'600000','start':args.event_start,'end':args.event_end,'adjust':adjust})
    specs.append({'id':'public_us','source':'yfinance'})
    results=[]
    context=mp.get_context('spawn')
    for spec in specs:
        q=context.Queue();job=context.Process(target=worker,args=(spec,str(out.root),q));job.start();job.join(45)
        if job.is_alive():
            job.terminate();job.join();result={**spec,'status':'timeout','timeout_seconds':45}
        else:
            try: result=q.get(timeout=2)
            except Empty: result={**spec,'status':'worker_failed','exit_code':job.exitcode}
        results.append(result);out.json('interface_results.json',{'versions':{n:importlib.metadata.version(n) for n in ('akshare','yfinance')},'results':results,'automatic_source_fallback':False,'attempts_per_request':1,'not_tested':{'tushare':'No authorized account token supplied','BigQuant':'No authorized platform session supplied'}})
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
