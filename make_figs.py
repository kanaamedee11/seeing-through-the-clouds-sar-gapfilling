import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300})
R=json.load(open("results/results.json")); FD="paper/figures/"
names={"10":"Tree","20":"Shrub","30":"Grass","40":"Crop","50":"Built","60":"Bare","90":"Wetland"}
# fig1: RMSE vs gap length
fig,ax=plt.subplots(figsize=(3.4,2.5)); Ls=[1,3,6]
sty={"B0_clim":("Climatology","#888888","o"),"B1_interp":("Linear interpolation","#c0504d","s"),"B2_blend":("Interp.+clim. blend","#e69f00","^"),"M1_optical":("ML, optical only","#1f4e79","D"),"M2_optical+SAR":("ML, optical + SAR","#2a9d8f","D"),"M3_SAR+clim":("ML, SAR + clim.","#6a3d9a","v")}
for m,(lab,c,mk) in sty.items(): ax.plot(Ls,[R["byL"][str(L)][m] for L in Ls],marker=mk,c=c,ms=3.5,label=lab,lw=1)
ax.set_xticks(Ls); ax.set_xlabel("Gap length (months)"); ax.set_ylabel("RMSE of NDVI"); ax.legend(frameon=False,fontsize=6,loc="upper left",ncol=2); ax.set_ylim(0.085,0.19)
fig.tight_layout(); fig.savefig(FD+"fig_rmse.pdf"); plt.close()
# fig2: SAR gain by class
fig,ax=plt.subplots(figsize=(3.4,2.5)); w=.26; cl=list(names)
for j,L in enumerate(Ls):
    g=[R["byClass"][f"{L}_{c}"]["gain_pct"] for c in cl]
    ax.bar(np.arange(7)+(j-1)*w,[x[0] for x in g],w,yerr=[[x[0]-x[1] for x in g],[x[2]-x[0] for x in g]],color=["#9ecae1","#4292c6","#08519c"][j],label=f"{L} month"+("s" if L>1 else ""),error_kw=dict(lw=.6,capsize=1))
ax.axhline(0,c="k",lw=.5); ax.set_xticks(range(7)); ax.set_xticklabels([names[c] for c in cl],fontsize=6.5); ax.set_ylabel("RMSE reduction from adding SAR (%)"); ax.legend(frameon=False,fontsize=6.5,title="Gap",title_fontsize=6.5)
fig.tight_layout(); fig.savefig(FD+"fig_gain.pdf"); plt.close()
# fig3: conformal coverage by class (L=3)
fig,axs=plt.subplots(1,2,figsize=(7.0,2.3)); c3=R["conformal"]["3"]["by_class"]
x=np.arange(7); w=.38
axs[0].bar(x-w/2,[c3[c]["cov_global"]*100 for c in cl],w,color="#c0504d",label="Global interval"); axs[0].bar(x+w/2,[c3[c]["cov_mondrian"]*100 for c in cl],w,color="#2a9d8f",label="Class-wise interval")
axs[0].axhline(90,c="k",ls="--",lw=.7); axs[0].set_ylim(70,106); axs[0].set_xticks(x); axs[0].set_xticklabels([names[c] for c in cl],fontsize=6.5); axs[0].set_ylabel("Empirical coverage (%)"); axs[0].legend(frameon=False,fontsize=6.5,loc="upper center",ncol=2); axs[0].set_title("Target 90%",fontsize=8)
axs[1].bar(x-w/2,[c3[c]["w_global"] for c in cl],w,color="#c0504d"); axs[1].bar(x+w/2,[c3[c]["w_mondrian"] for c in cl],w,color="#2a9d8f")
axs[1].set_xticks(x); axs[1].set_xticklabels([names[c] for c in cl],fontsize=6.5); axs[1].set_ylabel("Mean interval width (NDVI)"); axs[1].set_title("Width",fontsize=8)
fig.tight_layout(); fig.savefig(FD+"fig_conf.pdf"); plt.close()
# fig4: natural gaps + availability
P=pd.read_pickle("panel.pkl").sort_values(["pid","t"]); Y=P.ndvi.values.reshape(-1,84); runs=[]
for row in np.isnan(Y):
    c=0
    for v in row:
        if v: c+=1
        elif c: runs.append(c); c=0
    if c: runs.append(c)
runs=np.array(runs)
fig,axs=plt.subplots(1,2,figsize=(7.0,2.2))
vals=[(runs==k).sum() for k in range(1,8)]; axs[0].bar(range(1,8),vals,color="#1f4e79"); axs[0].set_yscale("log"); axs[0].set_xlabel("Length of natural gap (months)"); axs[0].set_ylabel("Number of gaps")
mm=P.assign(m=P.month).groupby("m").ndvi.apply(lambda s:s.isna().mean()*100); axs[1].bar(mm.index,mm.values,color="#e69f00"); axs[1].set_xlabel("Calendar month"); axs[1].set_ylabel("Missing monthly NDVI (%)"); axs[1].set_xticks(range(1,13))
fig.tight_layout(); fig.savefig(FD+"fig_gaps.pdf"); plt.close(); print("ok",mm.round(1).to_dict())
