import numpy as np, rasterio, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family":"serif","font.size":7,"savefig.dpi":300})
A="data/tile/"
Z=np.load("results_tile/tile_maps.npz"); wc=Z["wc"]; water=wc==80
tc=rasterio.open(A+"gapfill_tile_truecolour.tif").read().astype(float)
def rgb(a):
    x=np.moveaxis(a,0,-1); valid=x.sum(-1)>0; out=np.zeros_like(x)
    for k in range(3):
        lo,hi=np.percentile(x[...,k][valid],(2,98)); out[...,k]=np.clip((x[...,k]-lo)/(hi-lo),0,1)
    out[~valid]=1.0; return out**0.9
ext=[0,24.7,0,26.7]; ext20=[0,24.56,0,26.64]
fig,axs=plt.subplots(2,4,figsize=(7.2,4.0))
axs[0,0].imshow(rgb(tc[:3]),extent=ext20); axs[0,0].set_title("(a) Sentinel-2, Jul 2024",fontsize=7)
axs[0,1].imshow(rgb(tc[3:]),extent=ext20); axs[0,1].set_title("(b) Sentinel-2, Nov 2024\n(white: no clear view)",fontsize=7)
def show(ax,img,vmin,vmax,cmap,title,cb=True):
    im=np.ma.masked_invalid(img); bg=np.where(water,0.75,1.0); ax.imshow(bg,cmap="gray",vmin=0,vmax=1,extent=ext)
    h=ax.imshow(np.ma.masked_where(water,im),cmap=cmap,vmin=vmin,vmax=vmax,extent=ext); ax.set_title(title,fontsize=7)
    if cb: c=plt.colorbar(h,ax=ax,fraction=0.046,pad=0.02); c.ax.tick_params(labelsize=5.5)
show(axs[0,2],Z["truth"],0,0.9,"YlGn","(c) NDVI, Oct 2024 (truth)")
show(axs[0,3],Z["vv"],-16,-2,"gray","(d) Sentinel-1 VV, Oct 2024 (dB)")
nd=Z["B1_interp"]; truth=Z["truth"]
e1=np.abs(truth-Z["M1_optical"]); e2=np.abs(truth-Z["M2_optical+SAR"]); eb=np.abs(truth-Z["B1_interp"])
show(axs[1,0],Z["M2_optical+SAR"],0,0.9,"YlGn","(e) NDVI filled by M2")
show(axs[1,1],eb,0,0.4,"magma_r","(f) |Error|, interpolation")
show(axs[1,2],e2,0,0.4,"magma_r","(g) |Error|, optical + SAR")
show(axs[1,3],e1-e2,-0.15,0.15,"RdBu","(h) |Err| optical only minus\noptical + SAR (blue: SAR better)")
for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig("paper/figures/fig_tile.pdf"); print("ok")
