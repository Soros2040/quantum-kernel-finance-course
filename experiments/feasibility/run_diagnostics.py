"""Run a labelled raw-price diagnostic when total-return data admission is pending.

This is D0, not the approved total-return backtest. No 2025 records are admitted.
All input paths and private output roots must be supplied explicitly.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from kernel_models import preprocess, kernel_map, predict_mapped, explicit_ridge, rbf, krr
from quantum_core import state_features, fidelity_kernel, real_density_features
from safe_io import OutputRoot

FEATURES = ['momentum5','momentum20','volatility20','relative_volume20']
CANDIDATES = [('Q1',.5,1),('Q2',1.,1),('Q3',.5,2),('Q4',1.,2)]
LAMBDAS = [.001,.01,.1]


def panel(path, calendar_path=None):
    raw = pd.read_csv(path, dtype={'symbol':str}, parse_dates=['date'])
    if (raw.date.dt.year >= 2025).any():
        raise ValueError('Sealed or later records present in diagnostic input')
    if raw.duplicated(['symbol','date']).any():
        raise ValueError('Duplicate security-date keys')
    if calendar_path:
        calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(calendar_path)['date'])).sort_values().unique()
        calendar = calendar[(calendar.year>=2019)&(calendar.year<=2024)]
        calendar_origin = 'supplied calendar; provenance in data manifest'
    else:
        calendar = pd.DatetimeIndex(sorted(raw.date.unique()))
        calendar_origin = 'observed union, not certified exchange calendar'
    tables=[]
    for symbol, rows in raw.groupby('symbol', sort=True):
        rows = rows.set_index('date').reindex(calendar)
        op, close, volume = [pd.to_numeric(rows[k], errors='coerce').where(lambda v:v>0) for k in ('open','close','volume')]
        r = close / close.shift(1) - 1
        table = pd.DataFrame({'symbol':symbol,'date':calendar,'j':np.arange(len(calendar)),
                              'momentum5':(close/close.shift(5)-1).values,
                              'momentum20':(close/close.shift(20)-1).values,
                              'volatility20':r.rolling(20,min_periods=20).std(ddof=1).values,
                              'relative_volume20':np.log(volume/volume.shift(1).rolling(20,min_periods=20).mean()).values,
                              'y':(op.shift(-6)/op.shift(-1)-1).values,
                              'log_return':np.log(close/close.shift(1)).values})
        table['label_start']=table['j']+1
        table['label_end']=table['j']+6
        tables.append(table)
    frame=pd.concat(tables,ignore_index=True)
    frame['valid']=np.isfinite(frame[FEATURES+['y']]).all(axis=1)
    return frame,calendar,calendar_origin


def train_rows(frame, cutoff, n):
    eligible=frame[(frame.j<=cutoff)&(frame.j>=cutoff-251)&(frame.label_end<=cutoff)&frame.valid]
    bins=np.array_split(np.arange(max(0,cutoff-251),cutoff+1),n//4)
    counts={s:0 for s in frame.symbol.unique()}
    rng=np.random.default_rng(20261109+int(cutoff))
    take=[]
    for box in bins:
        subset=eligible[eligible.j.isin(box)]
        if subset.empty: continue
        daily=subset[subset.j==subset.j.min()].copy()
        daily['_count']=daily.symbol.map(counts)
        daily['_tie']=rng.random(len(daily))
        chosen=daily.sort_values(['_count','_tie','symbol']).head(4)
        take.extend(chosen.index.tolist())
        for s in chosen.symbol: counts[s]+=1
    return eligible.loc[take].sort_values(['j','symbol']),eligible


def grids(calendar,year):
    dates=np.flatnonzero(calendar.year==year)
    if not len(dates): return []
    return [int(j) for j in dates[::5] if j+6<len(calendar) and calendar[j+6].year==year]


def paired(frame,calendar,year):
    nstocks=frame.symbol.nunique()
    valid_dates=[]
    for j in grids(calendar,year):
        part=frame[(frame.j==j)&frame.valid]
        if len(part)==nstocks: valid_dates.append(j)
        if len(valid_dates)==2: break
    if len(valid_dates)!=2: raise ValueError('Two complete validation cross-sections unavailable')
    train,_=train_rows(frame,valid_dates[0]-1,64)
    evaluation=frame[frame.j.isin(valid_dates)&frame.valid].sort_values(['j','symbol'])
    if len(train)!=64 or len(evaluation)!=40: raise ValueError('64/40 admission failed')
    assert train.label_end.max()<evaluation.j.min()
    xt,xv,u,prep=preprocess(train[FEATURES].to_numpy(),evaluation[FEATURES].to_numpy(),train.y.to_numpy())
    v=(evaluation.y.to_numpy()-prep['y_mean'])/prep['y_sd']
    kernels=[]
    for name,a,depth in CANDIDATES:
        st=state_features(xt,a,depth);sv=state_features(xv,a,depth)
        kernels.append((fidelity_kernel(st),fidelity_kernel(sv,st)))
    return train,evaluation,u,v,kernels,prep


def sample_kernel(train,cross,shots,rng):
    n=len(train);ix=np.triu_indices(n,1)
    values=rng.binomial(shots,np.clip(train[ix],0,1))/shots
    sampled=np.eye(n);sampled[ix]=values;sampled[(ix[1],ix[0])]=values
    sampled_cross=rng.binomial(shots,np.clip(cross,0,1))/shots
    return sampled,sampled_cross


def select_from_counts(matrices,u,v,regularization):
    losses=[];diag=[]
    for train,cross in matrices:
        try:
            pred,info=krr(train,cross,u,regularization)
            loss=float(np.mean((pred-v)**2))
            if not np.isfinite(loss): raise ValueError('Nonfinite loss')
            losses.append(loss);diag.append(info)
        except (ValueError,np.linalg.LinAlgError) as exc:
            losses.append(float('inf'));diag.append({'failure':type(exc).__name__})
    minimum=min(losses)
    if not np.isfinite(minimum): return None,losses,diag
    # Q1,Q2 precede Q3,Q4: shallower first, then stable candidate ID.
    chosen=next(i for i,x in enumerate(losses) if x<=minimum+1e-12)
    return chosen,losses,diag


def finite_measurement(frame,calendar,out):
    began=time.perf_counter()
    ct,cv,cu,cy,ck,cp=paired(frame,calendar,2022)
    calibration_rng=np.random.default_rng(2026110901)
    calibration=[sample_kernel(k,b,128,calibration_rng) for k,b in ck]
    candidates=[]
    for lam in LAMBDAS:
        _,losses,_=select_from_counts(calibration,cu,cy,lam)
        candidates.append((float(np.mean(losses)),lam))
    valid=[x for x in candidates if np.isfinite(x[0])]
    if not valid: raise ValueError('Calibration failed for full candidate pool')
    best=min(x[0] for x in valid)
    regularization=max(lam for loss,lam in valid if loss<=best+1e-12)
    train,evaluation,u,v,kernels,prep=paired(frame,calendar,2024)
    exact=[]
    for k,b in kernels:
        prediction,_=krr(k,b,u,regularization)
        exact.append(float(np.mean((prediction-v)**2)))
    exact=np.array(exact)
    rows=[];pairs=[]
    for shots in (32,128,512):
        for repeat in range(20):
            rng=np.random.default_rng(np.random.SeedSequence([20261109,shots,repeat]))
            matrices=[sample_kernel(k,b,shots,rng) for k,b in kernels]
            selected,losses,diagnostics=select_from_counts(matrices,u,v,regularization)
            regret=None if selected is None else float(exact[selected]-exact.min())
            rows.append({'shots':shots,'repeat':repeat,'selected':None if selected is None else CANDIDATES[selected][0],
                         'regret':regret,'failed':selected is None,
                         'estimated_losses':json.dumps([x if np.isfinite(x) else None for x in losses]),
                         'negative_eigenvalues':sum(d.get('negative_eigenvalues',0) for d in diagnostics),
                         'sampled_model_excess_validation_loss':None if selected is None else losses[selected]-float(exact.min())})
            for i in range(4):
                for j in range(i+1,4):
                    if np.isfinite(losses[i]) and np.isfinite(losses[j]):
                        pairs.append({'shots':shots,'repeat':repeat,'pair':f'Q{i+1}-Q{j+1}',
                                      'estimated_gap':losses[i]-losses[j],'exact_gap':float(exact[i]-exact[j])})
    result=pd.DataFrame(rows);result.to_csv(out.path('finite_repeats.csv'),index=False)
    gaps=pd.DataFrame(pairs,columns=['shots','repeat','pair','estimated_gap','exact_gap']);gaps['gap_error']=gaps.estimated_gap-gaps.exact_gap
    gaps['sign_error']=np.where(gaps.exact_gap==0,np.nan,(gaps.estimated_gap*gaps.exact_gap<0).astype(float))
    gaps.to_csv(out.path('finite_pairs.csv'),index=False)
    summaries=[]
    for shots,group in result.groupby('shots'):
        freq=group.selected.value_counts().reindex([x[0] for x in CANDIDATES],fill_value=0)
        summary={'shots':int(shots),'repeats':len(group),'selection_counts':freq.to_dict(),
                 'no_selection':int(group.failed.sum()),'mean_regret':float(group.regret.mean()) if group.regret.notna().any() else None,
                 'max_regret':float(group.regret.max()) if group.regret.notna().any() else None,'mode_fraction':float(freq.max()/len(group)),
                 'pairwise_repeat_agreement':float((freq*(freq-1)).sum()/(len(group)*(len(group)-1))),
                 'mean_negative_eigenvalues':float(group.negative_eigenvalues.mean())}
        for delta in (.01,.02,.05):
            count=int(((group.regret>delta)|group.failed).sum());n=len(group);z=1.959963984540054
            center=(count/n+z*z/(2*n))/(1+z*z/n)
            width=z*np.sqrt(count/n*(1-count/n)/n+z*z/(4*n*n))/(1+z*z/n)
            summary[f'failure_delta_{delta}']={'count':count,'proportion':count/n,'wilson95':[center-width,center+width]}
        summaries.append(summary)
    for name,rows_ in [('calibration_train',ct),('calibration_validation',cv),('diagnostic_train',train),('diagnostic_validation',evaluation)]:
        rows_[['symbol','date','j','label_start','label_end']].to_csv(out.path(name+'.csv'),index=False)
    gap_summary=gaps.groupby(['shots','pair']).agg(exact_gap=('exact_gap','first'),bias=('gap_error','mean'),sd=('estimated_gap','std'),sign_error_rate=('sign_error','mean'),n_valid=('repeat','count'))
    pair_names=[f'Q{i+1}-Q{j+1}' for i in range(4) for j in range(i+1,4)]
    gap_summary=gap_summary.reindex(pd.MultiIndex.from_product([(32,128,512),pair_names],names=['shots','pair'])).reset_index()
    gap_summary['n_valid']=gap_summary.n_valid.fillna(0).astype(int)
    gap_summary['n_failed_pairs']=20-gap_summary.n_valid
    gap_summary['denominator']=20
    gap_summary['interpretation']='gap statistics conditional on both candidate fits succeeding'
    gap_summary.to_csv(out.path('finite_gap_summary.csv'),index=False)
    return {'scope':'D0 raw-price diagnostic; not approved total-return experiment',
            'calibration_lambda':regularization,'calibration_grid':[{ 'loss':x if np.isfinite(x) else None,'lambda':y} for x,y in candidates],
            'exact_losses':exact.tolist(),'nominal_shots':248348672,'real_cloud_shots':0,
            'binomial_sampling_blocks':1116544,'summaries':summaries,'seconds':time.perf_counter()-began,
            'calibration_preprocess':cp,'diagnostic_preprocess':prep,
            'oracle_isolation':'Selection function accepts sampled matrices, labels and fixed lambda only; exact scores appended afterwards'}


def financial(frame,calendar,out):
    began=time.perf_counter(); predictions=[];configs=[];equivalence=[]
    all_dates={y:grids(calendar,y) for y in (2022,2023,2024)}
    for year in (2022,2023,2024):
        for quarter in range(1,5):
            days=[j for j in all_dates[year] if calendar[j].quarter==quarter]
            if not days: continue
            train,full=train_rows(frame,min(days)-1,128)
            ev=frame[frame.j.isin(days)&frame.valid].sort_values(['j','symbol'])
            if len(train)<16 or ev.empty: continue
            assert train.label_end.max()<ev.j.min()
            xt,xv,u,prep=preprocess(train[FEATURES].to_numpy(),ev[FEATURES].to_numpy(),train.y.to_numpy())
            models=[('ridge',xt@xt.T,xv@xt.T,None,None)]
            for gamma in (.1,1.,10.): models.append((f'rbf_g{gamma}',rbf(xt,xt,gamma),rbf(xv,xt,gamma),None,None))
            for name,a,depth in CANDIDATES:
                st=state_features(xt,a,depth);sv=state_features(xv,a,depth)
                models.append((name,fidelity_kernel(st),fidelity_kernel(sv,st),st,sv))
            def append_prediction(name,lam,pred):
                target=ev[['symbol','date','j','y']].copy()
                target['year']=year;target['quarter']=quarter;target['model']=name;target['lambda']=lam
                target['prediction']=pred;predictions.append(target)
            for name,k,b,st,sv in models:
                zt,zv,info=kernel_map(k,b)
                for lam in LAMBDAS:
                    p=predict_mapped(zt,zv,u,lam)
                    append_prediction(name,lam,p*prep['y_sd']+prep['y_mean'])
                    if st is not None:
                        pe=explicit_ridge(real_density_features(st),real_density_features(sv),u,lam,True)
                        error=float(np.max(np.abs(p-pe)))
                        if not np.allclose(p,pe,atol=1e-8,rtol=1e-8): raise AssertionError('Matched feature / kernel predictions disagree')
                        equivalence.append({'year':year,'quarter':quarter,'candidate':name,'lambda':lam,'max_error':error})
                configs.append({'year':year,'quarter':quarter,'model':name,'n_train':len(train),'n_eval':len(ev),'train_cutoff':str(calendar[min(days)-1].date()),**info})
            xf,xfe,uf,pf=preprocess(full[FEATURES].to_numpy(),ev[FEATURES].to_numpy(),full.y.to_numpy())
            for lam in LAMBDAS:
                pred=explicit_ridge(xf,xfe,uf,lam)*pf['y_sd']+pf['y_mean']
                append_prediction('ridge_full',lam,pred)
            configs.append({'year':year,'quarter':quarter,'model':'ridge_full','n_train':len(full),'n_eval':len(ev),'train_cutoff':str(calendar[min(days)-1].date())})
            append_prediction('momentum20',np.nan,ev.momentum20.to_numpy())
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(out.path('development_predictions.csv'),index=False)
    daily=[]
    for (year,model,lam,date),g in pred.groupby(['year','model','lambda','date'],dropna=False):
        ic=np.nan
        if len(g)>=3 and g.prediction.nunique()>1 and g.y.nunique()>1:
            ic=float(np.corrcoef(rankdata(g.prediction),rankdata(g.y))[0,1])
        daily.append({'year':year,'model':model,'lambda':lam,'date':date,'count':len(g),'rank_ic':ic,
                      'mse':np.nan if model=='momentum20' else float(np.mean((g.prediction-g.y)**2))})
    daily=pd.DataFrame(daily);daily.to_csv(out.path('daily_prediction_metrics.csv'),index=False)
    summary=daily.groupby(['year','model','lambda'],dropna=False).agg(mse=('mse','mean'),rank_ic=('rank_ic','mean'),dates=('date','count'),records=('count','sum')).reset_index()
    summary.to_csv(out.path('development_metrics.csv'),index=False)
    pd.DataFrame(configs).to_csv(out.path('fit_configs.csv'),index=False)
    pd.DataFrame(equivalence).to_csv(out.path('feature_equivalence.csv'),index=False)
    return {'scope':'D0 unadjusted opening-price target, development selection evidence only',
            'prediction_rows':len(pred),'fit_configurations':len(configs),'quarters':len({(r['year'],r['quarter']) for r in configs}),
            'matched_feature_max_error':max(x['max_error'] for x in equivalence),
            'economic_backtest':'not run: total-return corporate-action and execution gates unverified',
            'sealed_2025_used':False,'seconds':time.perf_counter()-began}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--calendar');parser.add_argument('--output',required=True)
    args=parser.parse_args();out=OutputRoot(args.output)
    t=time.perf_counter();frame,calendar,origin=panel(args.input,args.calendar)
    report={'protocol_variant':'D0-v1','started_utc':datetime.now(timezone.utc).isoformat(),
            'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            'calendar_origin':origin,'symbols':frame.symbol.nunique(),'raw_calendar_rows':len(frame),
            'valid_supervised_rows':int(frame.valid.sum()),'G4_total_return_admitted':False,
            'financial_claims':'diagnostic only; no total-return, strategy-profit or quantum-advantage conclusion',
            'execution_location':os.environ.get('EXECUTION_LOCATION','unspecified')}
    for name,fn in [('finite_measurement',finite_measurement),('financial',financial)]:
        try: report[name]=fn(frame,calendar,out)
        except Exception as exc:
            report[name]={'error':type(exc).__name__,'message':str(exc)}
        out.json('diagnostic_summary.json',report)
    report['seconds']=time.perf_counter()-t
    out.json('diagnostic_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__': main()
