import sys, numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
import os
import exp as E, rasterio
from rasterio.warp import transform
os.makedirs("results_tile",exist_ok=True)
A="data/tile/"
# ---- 1. train on points OUTSIDE the tile (plus 0.1 deg margin), years 2019-2023, 3-month gaps
inside=(E.lon>29.18)&(E.lon<29.60)&(E.lat>-3.62)&(E.lat<-3.18); print("points excluded near tile:",int(inside.sum()))
rs=np.random.default_rng(11); pp,tt=np.where(E.obs); keep=(~inside[pp])&E.TRAIN_Y[tt]; itr=np.where(keep)[0]; itr=rs.choice(itr,250000,replace=False)
Xtr=E.build(pp[itr],tt[itr],3,rs); ytr=E.Y[pp[itr],tt[itr]]
models={n:E.fit(Xtr,ytr,c) for n,c in E.SETS.items() if n!="M3_SAR+clim"}
lr=E.LinearRegression().fit(Xtr[["interp","clim"]].fillna(0.4),ytr)
# ---- 2. tile arrays
m=rasterio.open("data/tile/gapfill_tile_monthly.tif"); a=m.read(); H,W=a.shape[1:]
nd=a[0::3].astype(float); nd[nd==-32768]=np.nan; vv=a[1::3].astype(float)/100; vh=a[2::3].astype(float)/100
st=rasterio.open(A+"gapfill_tile_static.tif").read().astype(float)
rain=pd.read_csv(A+"gapfill_tile_rain.csv").sort_values(["year","month"]).rain_mm.values
T=84; npx=H*W
Yt=(nd/1000).reshape(T,npx).T; VVt=np.clip(vv,-30,5).reshape(T,npx).T; VHt=np.clip(vh,-35,0).reshape(T,npx).T
RAINt=np.tile(rain[None,:],(npx,1))
rows,cols=np.divmod(np.arange(npx),W); xs=m.bounds.left+(cols+.5)*100; ys=m.bounds.top-(rows+.5)*100
lo,la=transform("EPSG:32735","EPSG:4326",xs.tolist(),ys.tolist())
E.Y,E.VV,E.VH,E.RAIN=Yt,VVt,VHt,RAINt
E.elev,E.slope,E.wc=st[1].ravel(),st[2].ravel(),st[0].ravel(); E.lat,E.lon=np.array(la),np.array(lo); E.npx=npx
E.obs=~np.isnan(Yt); idx=np.arange(T)
a_=np.where(E.obs,idx[None,:],-1); E.LO=np.maximum.accumulate(a_,axis=1)
b_=np.where(E.obs,idx[None,:],10**6); E.NO=np.minimum.accumulate(b_[:,::-1],axis=1)[:,::-1]
ysum=np.zeros((npx,12)); ycnt=np.zeros((npx,12))
for t in np.where(E.TRAIN_Y)[0]:
    o=E.obs[:,t]; ysum[o,E.month[t]-1]+=Yt[o,t]; ycnt[o,E.month[t]-1]+=1
E.ysum,E.ycnt=ysum,ycnt
# ---- 3. hide Sep-Nov 2024 (t = 68,69,70), target Oct 2024 (t=69)
TS=69; assert E.year[TS]==2024 and E.month[TS]==10
class Mid:
    def integers(self,a,b,size): return np.full(size,1)
land=st[0].ravel()!=80; p=np.where(E.obs[:,TS]&land)[0]
X=E.build(p,np.full(len(p),TS),3,Mid()); y=Yt[p,TS]
pred={"B0_clim":X.clim.values,"B1_interp":X.interp.values,"B2_blend":lr.predict(X[["interp","clim"]].fillna(0.4))}
for n,mo in models.items(): pred[n]=mo.predict(X[E.SETS[n]])
rm=lambda v: float(np.sqrt(np.mean((y-v)**2)))
out={"n_pixels":int(len(p)),"rmse":{k:rm(v) for k,v in pred.items()},"by_class":{}}
wcp=E.wc[p]
for c in np.unique(wcp):
    mk=wcp==c; out["by_class"][int(c)]={"n":int(mk.sum()),**{k:float(np.sqrt(np.mean((y[mk]-v[mk])**2))) for k,v in pred.items()}}
e1=np.abs(y-pred["M1_optical"]); e2=np.abs(y-pred["M2_optical+SAR"]); out["share_pixels_sar_better"]=float((e2<e1).mean())
out["gain_pct_M2_vs_M1"]=float((1-out["rmse"]["M2_optical+SAR"]/out["rmse"]["M1_optical"])*100)
out["gain_pct_M2_vs_B1"]=float((1-out["rmse"]["M2_optical+SAR"]/out["rmse"]["B1_interp"])*100)
json.dump(out,open("results_tile/tile_results.json","w"),indent=1); print(json.dumps(out,indent=1))
maps={k:np.full(npx,np.nan) for k in list(pred)+["truth"]}
for k,v in pred.items(): maps[k][p]=v
maps["truth"][p]=y
np.savez_compressed("results_tile/tile_maps.npz",H=H,W=W,wc=st[0],**{k:v.reshape(H,W) for k,v in maps.items()},vv=vv[TS],ndvi_obs=nd[TS]/1000)

# ---- 4. all target months 2024-2025 on the tile, 3-month window centred on the target
allr=[]
for TS in range(60,83):                      # t-1 and t+1 must exist; 2025-12 (83) has no later month
    p=np.where(E.obs[:,TS]&land)[0]
    X=E.build(p,np.full(len(p),TS),3,Mid()); y=Yt[p,TS]
    pr={"B0_clim":X.clim.values,"B1_interp":X.interp.values,"B2_blend":lr.predict(X[["interp","clim"]].fillna(0.4))}
    for n,mo in models.items(): pr[n]=mo.predict(X[E.SETS[n]])
    allr.append({"t":TS,"year":int(E.year[TS]),"month":int(E.month[TS]),"n":int(len(p)),**{k:float(np.sqrt(np.mean((y-v)**2))) for k,v in pr.items()}})
D=pd.DataFrame(allr); D.to_csv("results_tile/tile_allmonths.csv",index=False)
ms=["B0_clim","B1_interp","B2_blend","M1_optical","M2_optical+SAR"]
print(D[["year","month"]+ms].round(4).to_string(index=False))
print("mean RMSE",D[ms].mean().round(4).to_dict())
print("months best:",D[ms].idxmin(axis=1).value_counts().to_dict())
print("M2 < M1 in",int((D["M2_optical+SAR"]<D["M1_optical"]).sum()),"of",len(D),"; M2 < B1 in",int((D["M2_optical+SAR"]<D["B1_interp"]).sum()),"; M2 < B2 in",int((D["M2_optical+SAR"]<D["B2_blend"]).sum()))

# ---- 5. pooled by class over all months, and paired month-level statistics
acc={}
for TS in range(60,83):
    p=np.where(E.obs[:,TS]&land)[0]; X=E.build(p,np.full(len(p),TS),3,Mid()); y=Yt[p,TS]
    pr={"B1_interp":X.interp.values,"B2_blend":lr.predict(X[["interp","clim"]].fillna(0.4))}
    for n,mo in models.items(): pr[n]=mo.predict(X[E.SETS[n]])
    for c in np.unique(E.wc[p]):
        mk=E.wc[p]==c
        for k,v in pr.items():
            s=acc.setdefault((int(c),k),[0.0,0]); s[0]+=float(np.sum((y[mk]-v[mk])**2)); s[1]+=int(mk.sum())
cls={}
for (c,k),(s,n) in acc.items(): cls.setdefault(c,{})[k]=float(np.sqrt(s/n))
for c in cls: cls[c]["gain_M2_vs_M1_pct"]=(1-cls[c]["M2_optical+SAR"]/cls[c]["M1_optical"])*100
D["gain_M2_vs_M1"]=(1-D["M2_optical+SAR"]/D["M1_optical"])*100
rng=np.random.default_rng(2); bs=[D.gain_M2_vs_M1.values[rng.integers(0,len(D),len(D))].mean() for _ in range(5000)]
summ={"months":int(len(D)),"mean_rmse":{k:float(D[k].mean()) for k in ["B1_interp","B2_blend","M1_optical","M2_optical+SAR"]},
 "M2_vs_M1_mean_pct":float(D.gain_M2_vs_M1.mean()),"M2_vs_M1_ci":[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
 "M2_better_than_M1_months":int((D["M2_optical+SAR"]<D["M1_optical"]).sum()),"M2_better_than_B1_months":int((D["M2_optical+SAR"]<D["B1_interp"]).sum()),
 "M2_better_than_B2_months":int((D["M2_optical+SAR"]<D["B2_blend"]).sum()),
 "M2_vs_B1_mean_pct":float((1-D["M2_optical+SAR"].mean()/D["B1_interp"].mean())*100),"M1_vs_B2_mean_pct":float((1-D["M1_optical"].mean()/D["B2_blend"].mean())*100),
 "by_class_pooled":cls}
json.dump(summ,open("results_tile/tile_summary.json","w"),indent=1); print(json.dumps(summ,indent=1))

# ---- 6. gain expected from the point test, weighted by this tile's land-cover mix (3-month gaps)
P = json.load(open("results/results.json")); cl = [10, 20, 30, 40, 50, 60, 90]
cnt = np.array([(st[0] == c).sum() for c in cl], float); gn = np.array([P["byClass"][f"3_{c}"]["gain_pct"][0] for c in cl])
summ["expected_gain_from_point_test_pct"] = float((cnt / cnt.sum() * gn).sum())
json.dump(summ, open("results_tile/tile_summary.json", "w"), indent=1); print("expected tile gain from point test:", round(summ["expected_gain_from_point_test_pct"], 2))
