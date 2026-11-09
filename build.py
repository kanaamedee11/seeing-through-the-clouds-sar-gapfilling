import pandas as pd, numpy as np, glob, re
D="data/"
st=pd.read_csv(D+"gapfill_static.csv").drop(columns=["system:index"])
st["pid"]=np.arange(len(st)); key=lambda d:(d.lon.round(5).astype(str)+"_"+d.lat.round(5).astype(str))
st["k"]=key(st); print("static dup keys",st.k.duplicated().sum())
rows=[]
for y in range(2019,2026):
    d=pd.read_csv(D+f"gapfill_{y}.csv").drop(columns=["system:index"]); d["k"]=key(d)
    print(y,"dup",d.k.duplicated().sum(),"missing in static",(~d.k.isin(st.k)).sum())
    d=d.drop_duplicates("k")
    for m in range(1,13):
        t=f"_{y}_{m:02d}"; sub=pd.DataFrame({"k":d.k,"year":y,"month":m,"ndvi":d["ndvi"+t],"nobs":d["nobs"+t],"vv":d["vv"+t],"vh":d["vh"+t],"rain":d["rain"+t]})
        rows.append(sub)
P=pd.concat(rows,ignore_index=True)
for c in("ndvi","vv","vh"): P.loc[P[c]==-32768,c]=np.nan
P["ndvi"]/=1000; P["vv"]/=100; P["vh"]/=100
P=P.merge(st.drop_duplicates("k"),on="k",how="left")
P["t"]=(P.year-2019)*12+P.month-1
P.to_pickle("panel.pkl"); print(P.shape)
print(P[["ndvi","nobs","vv","vh","rain"]].describe().T)
print("share NDVI observed",P.ndvi.notna().mean(),"share S1 observed",P.vv.notna().mean())
print(P.groupby("wc").agg(n=("pid","nunique"),ndvi_obs=("ndvi",lambda s:s.notna().mean()),vv=("vv","mean"),ndvi=("ndvi","mean")).round(3))
print(P.groupby("month").ndvi.apply(lambda s:s.notna().mean()).round(2).to_dict())
print(P.groupby("year").agg(ndvi_obs=("ndvi",lambda s:s.notna().mean()),vv_obs=("vv",lambda s:s.notna().mean())).round(3))
