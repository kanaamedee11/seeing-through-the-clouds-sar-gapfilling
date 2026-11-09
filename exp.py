import numpy as np, pandas as pd, pickle, time, sys, warnings
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
P=pd.read_pickle("panel.pkl")
P["vv"]=P.vv.clip(-30,5); P["vh"]=P.vh.clip(-35,0)
npx=P.pid.nunique(); T=84
P=P.sort_values(["pid","t"]); assert len(P)==npx*T
def M(c): return P[c].values.reshape(npx,T)
Y=M("ndvi"); VV=M("vv"); VH=M("vh"); RAIN=M("rain").astype(float)
st=P.drop_duplicates("pid").sort_values("pid")
elev,slope,lat,lon,wc=[st[c].values for c in("elev_m","slope_deg","lat","lon","wc")]
# spatial blocks 0.25 deg
blk=(np.floor(lat/0.25).astype(int)*100+np.floor(lon/0.25).astype(int)); ub=np.unique(blk); print("blocks",len(ub))
rng=np.random.default_rng(3); perm=rng.permutation(ub); fold_of_block={b:i%5 for i,b in enumerate(perm)}; fold=np.array([fold_of_block[b] for b in blk])
obs=~np.isnan(Y)
idx=np.arange(T)
def last_obs():  # index of last observed <= t, -1 if none
    a=np.where(obs,idx[None,:],-1); return np.maximum.accumulate(a,axis=1)
def next_obs():
    a=np.where(obs,idx[None,:],10**6); return np.minimum.accumulate(a[:,::-1],axis=1)[:,::-1]
LO=last_obs(); NO=next_obs()
month=(idx%12)+1; year=idx//12+2019
TRAIN_Y=(year<=2023); TEST_Y=(year>=2024)
# climatology from train years: sum/count by pixel,month
ysum=np.zeros((npx,12)); ycnt=np.zeros((npx,12))
for t in np.where(TRAIN_Y)[0]:
    o=obs[:,t]; ysum[o,month[t]-1]+=Y[o,t]; ycnt[o,month[t]-1]+=1
def clim(p,t):
    m=month[t]-1; s=ysum[p,m].copy(); c=ycnt[p,m].copy()
    tr=TRAIN_Y[t]; own=np.where(obs[p,t]&tr,Y[p,t],0.0); s-=own; c-=(obs[p,t]&tr).astype(float)
    return np.where(c>0,s/np.maximum(c,1),np.nan)
def build(p,t,L,rs):
    off=rs.integers(0,L,size=len(p)); s=t-off; e=s+L-1
    sp=np.clip(s-1,0,T-1); ep=np.clip(e+1,0,T-1)
    ip=np.where(s-1>=0,LO[p,sp],-1); inx=np.where(e+1<=T-1,NO[p,ep],10**6)
    has_p=ip>=0; has_n=inx<T
    yp=np.where(has_p,Y[p,np.clip(ip,0,T-1)],np.nan); yn=np.where(has_n,Y[p,np.clip(inx,0,T-1)],np.nan)
    dp=np.where(has_p,t-ip,np.nan); dn=np.where(has_n,inx-t,np.nan)
    c=clim(p,t)
    interp=np.where(has_p&has_n,yp+(yn-yp)*dp/(dp+dn),np.where(has_p,yp,np.where(has_n,yn,c)))
    interp=np.where(np.isnan(interp),c,interp)
    cp=np.where(has_p,clim(p,np.clip(ip,0,T-1)),np.nan) # climatology at the neighbours' months (anomaly context)
    cn=np.where(has_n,clim(p,np.clip(inx,0,T-1)),np.nan)
    anom_p=yp-cp; anom_n=yn-cn
    tm=np.clip(t,1,T-2)
    F=dict(clim=c,interp=interp,yp=yp,yn=yn,dp=dp,dn=dn,anom_p=anom_p,anom_n=anom_n,
      vv=VV[p,t],vh=VH[p,t],vvmh=VV[p,t]-VH[p,t],vv_m1=VV[p,np.clip(t-1,0,T-1)],vv_p1=VV[p,np.clip(t+1,0,T-1)],
      vh_m1=VH[p,np.clip(t-1,0,T-1)],vh_p1=VH[p,np.clip(t+1,0,T-1)],
      rain=RAIN[p,t],rain1=RAIN[p,np.clip(t-1,0,T-1)],rain2=RAIN[p,np.clip(t-2,0,T-1)],
      msin=np.sin(2*np.pi*month[t]/12),mcos=np.cos(2*np.pi*month[t]/12),elev=elev[p],slope=slope[p],lat=lat[p],lon=lon[p],wc=wc[p])
    return pd.DataFrame(F)
OPT=["clim","interp","yp","yn","dp","dn","anom_p","anom_n"]
SAR=["vv","vh","vvmh","vv_m1","vv_p1","vh_m1","vh_p1"]
STAT=["rain","rain1","rain2","msin","mcos","elev","slope","lat","lon","wc"]
SETS={"M1_optical":OPT+STAT,"M2_optical+SAR":OPT+SAR+STAT,"M3_SAR+clim":["clim"]+SAR+STAT}
def fit(Xtr,ytr,cols):
    m=HGB(max_iter=250,learning_rate=0.08,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=1.0,random_state=0,categorical_features=[cols.index("wc")] if "wc" in cols else None)
    return m.fit(Xtr[cols],ytr)
def run(L,nmax_train=250000,seed=0):
    rs=np.random.default_rng(seed); out=[]
    pp,tt=np.where(obs)
    for f in range(5):
        trm=(fold[pp]!=f)&TRAIN_Y[tt]; tem=(fold[pp]==f)&TEST_Y[tt]
        itr=np.where(trm)[0]; itr=rs.choice(itr,min(nmax_train,len(itr)),replace=False); ite=np.where(tem)[0]
        Xtr=build(pp[itr],tt[itr],L,rs); ytr=Y[pp[itr],tt[itr]]; Xte=build(pp[ite],tt[ite],L,rs); yte=Y[pp[ite],tt[ite]]
        res=pd.DataFrame({"pid":pp[ite],"t":tt[ite],"fold":f,"L":L,"y":yte,"wc":Xte.wc.values,"B0_clim":Xte.clim.values,"B1_interp":Xte.interp.values})
        lr=LinearRegression().fit(Xtr[["interp","clim"]].fillna(0.4),ytr); res["B2_blend"]=lr.predict(Xte[["interp","clim"]].fillna(0.4))
        for name,cols in SETS.items():
            m=fit(Xtr,ytr,cols); res[name]=m.predict(Xte[cols])
        res["gapctx_next_dist"]=Xte.dn.values; res["gapctx_prev_dist"]=Xte.dp.values
        out.append(res); print("L",L,"fold",f,"n_test",len(res),flush=True)
    return pd.concat(out,ignore_index=True)
if __name__=="__main__":
    t0=time.time(); R=[]
    for L in [int(x) for x in sys.argv[1:]]:
        R.append(run(L)); print("done L",L,time.time()-t0,flush=True)
    pd.concat(R).to_pickle("preds_"+"_".join(sys.argv[1:])+".pkl")
