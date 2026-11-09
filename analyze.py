import pandas as pd, numpy as np, json, glob
R=pd.concat([pd.read_pickle(f) for f in ["preds_1.pkl","preds_3_6.pkl"]],ignore_index=True)
R["month"]=(R.t%12)+1; R["year"]=R.t//12+2019
ms=["B0_clim","B1_interp","B2_blend","M1_optical","M2_optical+SAR","M3_SAR+clim"]
for m in ms: R["e2_"+m]=(R.y-R[m])**2; R["ae_"+m]=(R.y-R[m]).abs()
out={}
pids=np.sort(R.pid.unique()); rng=np.random.default_rng(5)
def cluster_boot(d,fn,B=1000):
    g=d.groupby("pid"); S=g.apply(lambda x: pd.Series({c:x[c].sum() for c in x.columns if c.startswith("e2_")}|{"n":len(x)}))
    ids=S.index.values; vals=S.values; cols=list(S.columns); est=[]
    for _ in range(B):
        s=rng.integers(0,len(ids),len(ids)); sm=vals[s].sum(0); est.append(fn(dict(zip(cols,sm))))
    est=np.array(est); return est
def rmse(sm,m): return np.sqrt(sm["e2_"+m]/sm["n"])
# 1 by L
res={}
for L,d in R.groupby("L"):
    r={m:float(np.sqrt(d["e2_"+m].mean())) for m in ms}
    r["mae"]={m:float(d["ae_"+m].mean()) for m in ms}
    r["r2"]={m:float(1-((d.y-d[m])**2).sum()/((d.y-d.y.mean())**2).sum()) for m in ms}
    def rel(a,b): return lambda sm:(1-rmse(sm,a)/rmse(sm,b))*100
    cmp={}
    for a,b in [("M2_optical+SAR","M1_optical"),("M1_optical","B1_interp"),("M2_optical+SAR","B1_interp"),("M3_SAR+clim","B0_clim"),("M2_optical+SAR","B2_blend")]:
        est=cluster_boot(d,rel(a,b)); cmp[f"{a}_vs_{b}_pct"]=[float(np.mean(est)),float(np.percentile(est,2.5)),float(np.percentile(est,97.5))]
    r["improvement_pct"]=cmp; r["n"]=int(len(d)); res[int(L)]=r
out["byL"]=res
# 2 by class, month, year (relative RMSE improvement M2 vs M1) per L
cls={}
for (L,w),d in R.groupby(["L","wc"]):
    cls[f"{L}_{int(w)}"]={"M1":float(np.sqrt(d["e2_M1_optical"].mean())),"M2":float(np.sqrt(d["e2_M2_optical+SAR"].mean())),"B1":float(np.sqrt(d["e2_B1_interp"].mean())),"B0":float(np.sqrt(d["e2_B0_clim"].mean())),"M3":float(np.sqrt(d["e2_M3_SAR+clim"].mean()))}
    est=cluster_boot(d,lambda sm:(1-rmse(sm,"M2_optical+SAR")/rmse(sm,"M1_optical"))*100,B=400); cls[f"{L}_{int(w)}"]["gain_pct"]=[float(np.mean(est)),float(np.percentile(est,2.5)),float(np.percentile(est,97.5))]
out["byClass"]=cls
mon={}
for (L,m),d in R.groupby(["L","month"]): mon[f"{L}_{m}"]=float((1-np.sqrt(d["e2_M2_optical+SAR"].mean())/np.sqrt(d["e2_M1_optical"].mean()))*100)
out["byMonth_gain_pct"]=mon
yr={}
for (L,y),d in R.groupby(["L","year"]): yr[f"{L}_{y}"]={m:float(np.sqrt(d["e2_"+m].mean())) for m in ms}
out["byYear"]=yr
# 3 split conformal (cross-fold, absolute residual, 90%), global vs class-conditional, for M2
alpha=0.1; conf={}
for L,d in R.groupby("L"):
    d=d.copy(); d["res"]=(d.y-d["M2_optical+SAR"]).abs(); cov_g=np.zeros(len(d),bool); wid_g=np.zeros(len(d)); cov_c=np.zeros(len(d),bool); wid_c=np.zeros(len(d))
    for f in range(5):
        cal=d[d.fold!=f]; te=(d.fold==f).values
        q=np.quantile(cal.res,np.ceil((len(cal)+1)*(1-alpha))/len(cal)); cov_g[te]=d.res.values[te]<=q; wid_g[te]=2*q
        for w in d.wc.unique():
            c=cal[cal.wc==w]; qw=np.quantile(c.res,min(1,np.ceil((len(c)+1)*(1-alpha))/len(c))); mk=te&(d.wc==w).values; cov_c[mk]=d.res.values[mk]<=qw; wid_c[mk]=2*qw
    d["cov_g"]=cov_g; d["cov_c"]=cov_c; d["w_g"]=wid_g; d["w_c"]=wid_c
    conf[int(L)]={"cov_global":float(d.cov_g.mean()),"cov_mondrian":float(d.cov_c.mean()),"width_global":float(d.w_g.mean()),"width_mondrian":float(d.w_c.mean()),
      "by_class":{int(w):{"cov_global":float(x.cov_g.mean()),"cov_mondrian":float(x.cov_c.mean()),"w_global":float(x.w_g.mean()),"w_mondrian":float(x.w_c.mean())} for w,x in d.groupby("wc")},
      "by_month_cov_global":{int(m):float(x.cov_g.mean()) for m,x in d.groupby("month")}}
out["conformal"]=conf
json.dump(out,open("results/results.json","w"),indent=1); R.to_pickle("allpreds.pkl")
# natural gaps
P=pd.read_pickle("panel.pkl").sort_values(["pid","t"]); Y=P.ndvi.values.reshape(-1,84); runs=[]
for row in np.isnan(Y):
    c=0
    for v in row:
        if v: c+=1
        elif c: runs.append(c); c=0
    if c: runs.append(c)
runs=np.array(runs); out2={"n_gaps":int(len(runs)),"share_len1":float((runs==1).mean()),"share_len2":float((runs==2).mean()),"share_ge3":float((runs>=3).mean()),"share_ge6":float((runs>=6).mean()),"max":int(runs.max()),"missing_share":float(np.isnan(Y).mean())}
out["naturalgaps"]=out2; json.dump(out,open("results/results.json","w"),indent=1)
print(json.dumps({k:out[k] for k in("byL","naturalgaps")},indent=1)); print(json.dumps(out["conformal"],indent=1)[:2500])
