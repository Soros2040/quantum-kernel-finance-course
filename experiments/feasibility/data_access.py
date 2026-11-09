"""Read-only local market-data audit. Outputs must be outside the source tree.

This module never executes provider code, loads pickle/configuration files,
extracts archives, changes permissions, or calls network market-data APIs.
Data values and source locations belong in private outputs, not this source.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, os, re, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
from safe_io import OutputRoot

BLOCKED = {"code", "real_trading", ".git", ".ssh"}
FIELDS = {"股票代码":"symbol", "股票名称":"name", "交易日期":"date", "开盘价":"open", "最高价":"high", "最低价":"low", "收盘价":"close", "前收盘价":"prev_close", "成交量":"volume", "成交额":"amount", "流通市值":"float_market_cap", "总市值":"total_market_cap"}
REQUIRED = ["symbol","date","open","high","low","close","prev_close","volume","amount"]

def stamp():
    return datetime.now(timezone.utc).isoformat()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def secure_output(source, output):
    guard = OutputRoot(output, source)
    return guard.source, guard.root

def stat_dict(path):
    s=path.stat()
    return {"size":s.st_size,"mtime_ns":s.st_mtime_ns}

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def reader(path):
    # Most provider CSVs have one descriptive line; index files start at the header.
    with path.open("rb") as probe:head=b"\n".join(probe.read(4096).splitlines()[:2])
    try:head.decode("utf-8-sig");encoding="utf-8-sig"
    except UnicodeDecodeError:encoding="gb18030"
    f=path.open("r",encoding=encoding,errors="strict",newline="")
    first=f.readline()
    cells=next(csv.reader([first]))
    known={"股票代码","交易日期","日期","date","trade_date","candle_end_time","timestamp","symbol"}
    if any(x.strip().lstrip("\ufeff") in known for x in cells):f.seek(0)
    return f, csv.DictReader(f)

def safe_date(value):
    value=(value or "").strip().replace("/","-")[:10]
    try:return datetime.strptime(value,"%Y-%m-%d").date().isoformat()
    except ValueError:return None

def day_files(source):
    with os.scandir(source/"stock-trading-data") as it:
        return sorted([Path(e.path) for e in it if e.is_file(follow_symlinks=False) and re.fullmatch(r"(?:sh|sz|bj)\d{6}\.csv",e.name)],key=lambda p:p.name)

def prepare(source, output, count, start, end, anchor):
    t0=time.perf_counter();files=day_files(source);selected=[];selection=[]
    for path in files:
        if len(selected)>=count:break
        found=False;first=None;scanned=0
        f,rows=reader(path)
        with f:
            for row in rows:
                d=safe_date(row.get("交易日期"));scanned+=1
                if d and first is None:first=d
                if d==anchor:found=True;break
                # Only the dated presence criterion is used, never later coverage.
                if d and d>anchor:break
        selection.append({"file":path.name,"first_date_observed":first,"anchor_present":found,"rows_checked":scanned})
        if found:selected.append(path)
    if len(selected)!=count:raise RuntimeError("Insufficient files satisfying the fixed anchor date")
    data=output/"data";data.mkdir(exist_ok=True)
    manifest={"created_utc":stamp(),"selection_rule":"sort file identifier, retain first files with anchor date; no later completeness screen", "anchor":anchor,"start":start,"end":end,"fixed_set_not_unbiased_historical_universe":True,"selected_symbols":[p.stem for p in selected],"files":[],"G4_verified":False,"allowed_interpretation":"raw-price diagnostic only; no total-return or economic-performance claim"}
    all_rows=[];num_errors=[]
    for path in selected:
        before=stat_dict(path);h_before=sha(path);f,rows=reader(path);kept=[];invalid_dates=0
        with f:
            raw_fields=rows.fieldnames or []
            mapped=[FIELDS.get(x,"raw_"+x) for x in raw_fields]
            missing=set(REQUIRED)-set(mapped)
            if missing:raise ValueError("Missing required fields: "+str(sorted(missing)))
            for row in rows:
                d=safe_date(row.get("交易日期"))
                if not d:invalid_dates+=1;continue
                if not start<=d<=end:continue
                item={FIELDS.get(k,"raw_"+k):v for k,v in row.items() if k is not None}
                item["date"]=d
                for key in ["open","high","low","close","prev_close","volume","amount","float_market_cap","total_market_cap"]:
                    if key in item:
                        try:item[key]=float(item[key])
                        except (ValueError,TypeError):num_errors.append({"symbol":path.stem,"date":d,"field":key});item[key]=""
                kept.append(item)
        after=stat_dict(path);h_after=sha(path);final=stat_dict(path)
        stable=before==after==final and h_before==h_after
        if not stable:raise RuntimeError("Source changed during read: "+path.name)
        all_rows.extend(kept)
        target=data/(path.stem+"_daily.csv")
        with target.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=mapped);w.writeheader();w.writerows(kept)
        manifest["files"].append({"source":str(path),"output":str(target),"raw_fields":raw_fields,"standard_fields":mapped,"before":before,"after":final,"sha256_before":h_before,"sha256_after":h_after,"source_stable":stable,"retained_rows":len(kept),"min_date":min((r['date'] for r in kept),default=None),"max_date":max((r['date'] for r in kept),default=None),"invalid_dates":invalid_dates})
    all_rows.sort(key=lambda r:(r["date"],r["symbol"]))
    columns=manifest["files"][0]["standard_fields"]
    combined=data/"daily_panel.csv"
    with combined.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(all_rows)
    manifest.update({"combined_path":str(combined),"combined_sha256":sha(combined),"rows":len(all_rows),"numeric_parse_errors":num_errors,"elapsed_seconds":time.perf_counter()-t0})
    dump(output/"inventory_selection.json",{"selection":selection,"files_total":len(files),"chosen":manifest["selected_symbols"]})
    dump(data/"manifest.json",manifest)
    print(json.dumps({k:manifest[k] for k in ["selected_symbols","rows","combined_path","G4_verified","elapsed_seconds"]},ensure_ascii=False),flush=True)
    return manifest

def sample_schema(path):
    result={"file":str(path),"coverage_status":"schema and bounded initial dates only; not full coverage"}
    try:
        f,rows=reader(path)
        with f:
            result["fields"]=rows.fieldnames
            observed=[];examined=0
            datekeys=[k for k in rows.fieldnames or [] if any(t in k.lower() for t in ["日期","date","time","时间"])]
            for row in rows:
                examined+=1
                for key in datekeys[:4]:
                    d=safe_date(row.get(key))
                    if d and d<"2025-01-01":observed.append(d)
                if examined>=64:break
            result.update({"rows_examined_bound":examined,"initial_dates_before_2025_min":min(observed,default=None),"initial_dates_before_2025_max":max(observed,default=None)})
    except Exception as e:result["error"]=type(e).__name__+": "+str(e)
    return result

def inventory(source,output):
    t0=time.perf_counter();result=[]
    with os.scandir(source) as it:top=sorted(list(it),key=lambda e:e.name)
    for entry in top:
        record={"name":entry.name,"type":"directory" if entry.is_dir(follow_symlinks=False) else "file"}
        if entry.name in BLOCKED:
            record["status"]="excluded: configuration/provider-code area; no descendants read";result.append(record);continue
        if not entry.is_dir(follow_symlinks=False):
            st=entry.stat(follow_symlinks=False);record.update({"bytes":st.st_size,"format":Path(entry.name).suffix.lower(),"status":"metadata only; archives not opened"});result.append(record);continue
        cap=500 if any(k in entry.name.lower() for k in ["min_","-5m","-15m","-1h","zip","coin"]) else 12000
        counts=collections.Counter();sizes=0;dirs=0;n=0;truncated=False;sample=None;child=None;begin=time.perf_counter()
        try:
            with os.scandir(entry.path) as it:
                for e in it:
                    if n>=cap or time.perf_counter()-begin>12:truncated=True;break
                    n+=1
                    if e.is_dir(follow_symlinks=False):dirs+=1;child=child or (Path(e.path) if e.name not in BLOCKED else None)
                    elif e.is_file(follow_symlinks=False):
                        ext=Path(e.name).suffix.lower();counts[ext]+=1;sizes+=e.stat(follow_symlinks=False).st_size
                        if sample is None and ext==".csv" and not any(k in e.name.lower() for k in ["credential","password","secret","token","account","config"]):sample=Path(e.path)
            record.update({"direct_entries_observed":n,"direct_files_observed":sum(counts.values()),"direct_subdirectories_observed":dirs,"formats_observed":dict(counts),"direct_file_bytes_observed":sizes,"metadata_complete_for_direct_children":not truncated,"recursive_count_performed":False,"entry_cap":cap,"elapsed_seconds":time.perf_counter()-begin})
            if sample is None and child is not None:
                with os.scandir(child) as it:
                    for j,e in enumerate(it):
                        if j>=100:break
                        if e.is_file(follow_symlinks=False) and Path(e.name).suffix.lower()==".csv":sample=Path(e.path);break
            if sample:record["schema_sample"]=sample_schema(sample)
            else:record["coverage_status"]="no CSV sampled; no content/coverage inference from directory names"
        except Exception as e:record["error"]=type(e).__name__+": "+str(e)
        result.append(record)
        dump(output/"inventory_top_level.json",{"created_utc":stamp(),"source":str(source),"elapsed_seconds":time.perf_counter()-t0,"entries":result,"all_top_level_done":False})
        print(json.dumps({"inventory":entry.name,"observed":n,"truncated":truncated},ensure_ascii=False),flush=True)
    final={"created_utc":stamp(),"source":str(source),"elapsed_seconds":time.perf_counter()-t0,"entries":result,"all_top_level_done":True}
    dump(output/"inventory_top_level.json",final)
    with (output/"inventory_top_level.csv").open("w",encoding="utf-8-sig",newline="") as f:
        cols=["name","type","direct_entries_observed","direct_files_observed","direct_subdirectories_observed","direct_file_bytes_observed","metadata_complete_for_direct_children","recursive_count_performed","formats_observed","status","coverage_status"]
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader()
        for r in result:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in r.items()})

def company_actions(source,output,start,end):
    manifest=json.loads((output/"data/manifest.json").read_text(encoding="utf-8"));entries=[];calendar=[]
    for symbol in manifest["selected_symbols"]:
        for directory,kind in [("stock-dividend-delivery","actions"),("stock-trading-date","security_dates")]:
            p=source/directory/(symbol+".csv")
            if not p.exists():entries.append({"symbol":symbol,"kind":kind,"status":"not_found"});continue
            before=stat_dict(p);h=sha(p);f,rows=reader(p);kept=[]
            with f:
                fields=rows.fieldnames or []
                for r in rows:
                    if kind=="security_dates":
                        d=safe_date(r.get("交易日期"))
                        if d and start<=d<=end:kept.append(r);calendar.append({"symbol":symbol,"date":d})
                    else:
                        # Require dated implementation/ex-event information inside the allowed interval.
                        dates=[safe_date(r.get(k)) for k in ["实施公告日","股权登记日","除权除息日","派息日","红股上市日"]]
                        if any(d and start<=d<=end for d in dates) and not any(d and d>end for d in dates):kept.append(r)
            after=stat_dict(p);h_after=sha(p);stable=before==after and h==h_after
            if not stable:raise RuntimeError("Source changed: "+p.name)
            target=output/"data"/(symbol+"_"+kind+".csv")
            with target.open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(kept)
            entries.append({"symbol":symbol,"kind":kind,"source":str(p),"output":str(target),"fields":fields,"rows":len(kept),"sha256_before":h,"sha256_after":h_after,"before":before,"after":after,"source_stable":stable})
    with (output/"data/security_dates.csv").open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=["symbol","date"]);w.writeheader();w.writerows(calendar)
    report={"created_utc":stamp(),"entries":entries,"G4_verified":False,"calendar_scope":"per-security observed trading dates; not certified exchange-wide historical calendar","limitations":["No verified provider data dictionary or adjustment-factor series","Gross cash dividends do not establish investor-specific after-tax total return","Event completeness, split/share delivery and delisting treatment unverified","Raw close versus next reported previous close is a diagnostic, not a certified wealth index"]}
    dump(output/"inventory_company_actions.json",report)
    print(json.dumps({"company_records":len(entries),"security_date_rows":len(calendar),"G4_verified":False}),flush=True)

def peak_rss():
    if os.name != "nt":return None
    try:
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
        pmc=PMC();pmc.cb=ctypes.sizeof(pmc)
        get=ctypes.windll.kernel32.GetCurrentProcess;get.restype=wintypes.HANDLE
        fn=ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD];fn.restype=wintypes.BOOL
        ok=fn(get(),ctypes.byref(pmc),pmc.cb)
        return int(pmc.PeakWorkingSetSize) if ok else None
    except Exception:return None

def worker(source,n,start,end):
    files=day_files(source);files=files if n==0 else files[:n];t0=time.perf_counter();retained=0;readrows=0;bytes_=0;errors=[]
    for p in files:
        bytes_+=p.stat().st_size
        try:
            # Only the date field is inspected; future rows never enter model data.
            with p.open("r",encoding="gb18030",errors="strict",newline="") as f:
                f.readline();header=next(csv.reader([f.readline()]));idx=header.index("交易日期")
                for line in f:
                    parts=line.rstrip("\r\n").split(",")
                    readrows+=1
                    if len(parts)>idx and start<=parts[idx]<=end:retained+=1
        except Exception as e:errors.append({"file":p.name,"error":type(e).__name__})
    return {"files":len(files),"source_bytes":bytes_,"peak_rss_bytes":peak_rss(),"rows_streamed":readrows,"rows_in_allowed_dates":retained,"elapsed_seconds":time.perf_counter()-t0,"errors":errors,"operation":"sequential date-only CSV stream aggregation; no full-universe DataFrame; OS cache not flushed"}

def benchmark(source,output,start,end):
    results=[];stop=False
    for n in [20,100,500,0]:
        point_start=time.perf_counter()
        for rep in range(1,4):
            remaining=300-(time.perf_counter()-point_start)
            if remaining<=0:results.append({"requested":n or "all","rep":rep,"status":"point_deadline"});stop=True;break
            command=[sys.executable,__file__,"worker","--source",str(source),"--count",str(n),"--start",start,"--end",end]
            try:
                call_start=time.perf_counter()
                p=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",timeout=remaining,env={**os.environ,"PYTHONIOENCODING":"utf-8"})
                parsed=[]
                for line in p.stdout.splitlines():
                    try:
                        obj=json.loads(line)
                        if isinstance(obj,dict) and "elapsed_seconds" in obj:parsed.append(obj)
                    except json.JSONDecodeError:pass
                value=parsed[-1];value["subprocess_wall_seconds"]=time.perf_counter()-call_start;value.update({"requested":n or "all","rep":rep,"status":"ok" if p.returncode==0 else "error"});results.append(value)
            except subprocess.TimeoutExpired:results.append({"requested":n or "all","rep":rep,"status":"timeout","point_limit_seconds":300});stop=True
            except Exception as e:results.append({"requested":n or "all","rep":rep,"status":"error","error":str(e)});stop=True
            dump(output/"inventory_read_benchmark.json",{"created_utc":stamp(),"point_total_limit_seconds":300,"repetitions_per_point":3,"results":results,"future_values_used":False})
            print(json.dumps(results[-1],ensure_ascii=False),flush=True)
            if stop:break
        if stop:break

def reconcile(source,output,start,end):
    data=output/"data"
    with (data/"daily_panel.csv").open(encoding="utf-8-sig",newline="") as f:panel=list(csv.DictReader(f))
    symbols=sorted({r["symbol"] for r in panel});by=collections.defaultdict(list)
    for r in panel:by[r["symbol"]].append(r)
    for rows in by.values():rows.sort(key=lambda r:r["date"])
    union={r["date"] for r in panel};report={"created_utc":stamp(),"G4_verified":False}
    index=source/"stock-main-index-data"/"sh000001.csv"
    if index.exists():
        before=stat_dict(index);h=sha(index);f,rows=reader(index);dates=[]
        with f:
            fields=rows.fieldnames or []
            date_key=next((k for k in ["交易日期","日期","date","candle_end_time"] if k in fields),None)
            if date_key is None:raise ValueError("Index date field not found")
            for row in rows:
                d=safe_date(row.get(date_key))
                if d and start<=d<=end:dates.append(d)
        after=stat_dict(index);h_after=sha(index)
        if before!=after or h!=h_after:raise RuntimeError("Index source changed during read")
        dates=sorted(set(dates))
        with (data/"market_calendar.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["date","source_symbol"]);w.writeheader();w.writerows({"date":d,"source_symbol":"sh000001"} for d in dates)
        report["calendar"]={"source":str(index),"date_field":date_key,"raw_fields":fields,"sha256_before":h,"sha256_after":h_after,"before":before,"after":after,"dates":len(dates),"first":min(dates,default=None),"last":max(dates,default=None),"stock_union_dates":len(union),"in_index_not_stock_union":sorted(set(dates)-union),"in_stock_union_not_index":sorted(union-set(dates)),"interpretation":"observed Shanghai Composite index dates; empirical market-calendar proxy, not exchange-issued calendar"}
    else:report["calendar"]={"status":"index_file_not_found","interpretation":"only per-security dates available"}
    events=[];jumps=[]
    for symbol in symbols:
        rows=by[symbol];lookup={r["date"]:i for i,r in enumerate(rows)}
        for prev,row in zip(rows,rows[1:]):
            lag=float(prev["close"]);pc=float(row["prev_close"])
            if abs(lag-pc)>.005:jumps.append({"symbol":symbol,"date":row["date"],"previous_observed_date":prev["date"],"previous_raw_close":lag,"reported_prev_close":pc,"difference":pc-lag})
        ap=data/(symbol+"_actions.csv")
        if not ap.exists():continue
        with ap.open(encoding="utf-8-sig",newline="") as f:actions=list(csv.DictReader(f))
        for action in actions:
            d=safe_date(action.get("除权除息日"));i=lookup.get(d)
            if i is None or i==0 or "实施" not in action.get("实施进度",""):continue
            def number(x):
                try:return float(x)
                except (TypeError,ValueError):return 0.0 if x in ["--","",None] else None
            cash=number(action.get("每股派息_税前"));bonus=number(action.get("送股比例"));capital=number(action.get("转增比例"))
            if None in [cash,bonus,capital]:continue
            prev=rows[i-1];current=rows[i];lag=float(prev["close"]);reported=float(current["prev_close"])
            expected=(lag-cash)/(1+bonus+capital)
            events.append({"symbol":symbol,"ex_date":d,"action_row":action,"previous_price_row":prev,"ex_date_price_row":current,"gross_cash_per_share":cash,"bonus_ratio_as_reported":bonus,"capital_ratio_as_reported":capital,"candidate_ex_reference":expected,"reported_prev_close":reported,"absolute_difference":abs(expected-reported),"formula_assumption":"(previous raw close - pre-tax cash)/(1+bonus+capital); excludes rights issues and other adjustments; -- treated as zero for this diagnostic only","not_total_return_validation":True})
    grouped=collections.defaultdict(list)
    for event in events:grouped[(event["symbol"],event["ex_date"])].append(event)
    multiple=[]
    for (symbol,d),group in grouped.items():
        if len(group)<2:continue
        cash=sum(e["gross_cash_per_share"] for e in group)
        bonus=sum(e["bonus_ratio_as_reported"] for e in group)
        capital=sum(e["capital_ratio_as_reported"] for e in group)
        expected=(float(group[0]["previous_price_row"]["close"])-cash)/(1+bonus+capital)
        multiple.append({"symbol":symbol,"ex_date":d,"number_of_rows":len(group),"action_rows":[e["action_row"] for e in group],"gross_cash_sum":cash,"candidate_ex_reference":expected,"reported_prev_close":group[0]["reported_prev_close"],"absolute_difference":abs(expected-group[0]["reported_prev_close"]),"interpretation":"same-day additive-event hypothesis; manual event identity review required, not automatic deduplication or total-return validation"})
    dump(data/"same_day_event_checks.json",{"groups":multiple,"G4_verified":False})
    report["same_day_multiple_action_groups"]=len(multiple)
    report.update({"raw_previous_close_discontinuities":len(jumps),"diagnostic_event_count":len(events),"diagnostic_events_matching_one_cent":sum(e["absolute_difference"]<=.01000001 for e in events),"limitations":["One-event arithmetic does not establish complete corporate-action coverage or a total-return series","Tax, entitlements, delivery dates, delisting, rights and adjustments remain unverified","Zeros substituted for -- only in diagnostic candidate formula; provider semantics not certified"]})
    dump(output/"inventory_calendar_and_adjustment.json",report)
    dump(data/"corporate_action_diagnostics.json",{"events":events,"raw_previous_close_jumps":jumps,"G4_verified":False})
    dump(data/"single_event_check.json",next((e for e in events if e["absolute_difference"]<=.01000001),{"status":"no_matching_event"}))
    print(json.dumps({"calendar":report["calendar"],"diagnostic_event_count":len(events),"matching_one_cent":report["diagnostic_events_matching_one_cent"],"G4_verified":False},ensure_ascii=False),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument("command",choices=["prepare","inventory","actions","benchmark","worker","reconcile"]);a.add_argument("--source",type=Path,required=True);a.add_argument("--output",type=Path);a.add_argument("--count",type=int,default=20);a.add_argument("--start",default="2019-01-01");a.add_argument("--end",default="2024-12-31");a.add_argument("--anchor",default="2019-01-02");args=a.parse_args()
    if args.end>"2024-12-31":raise ValueError("This development protocol excludes later years")
    if args.command=="worker":print(json.dumps(worker(args.source,args.count,args.start,args.end),ensure_ascii=False));return
    if args.output is None:a.error("--output is required")
    source,output=secure_output(args.source,args.output)
    if args.command=="prepare":prepare(source,output,args.count,args.start,args.end,args.anchor)
    elif args.command=="inventory":inventory(source,output)
    elif args.command=="actions":company_actions(source,output,args.start,args.end)
    elif args.command=="benchmark":benchmark(source,output,args.start,args.end)
    elif args.command=="reconcile":reconcile(source,output,args.start,args.end)

if __name__=="__main__":main()
