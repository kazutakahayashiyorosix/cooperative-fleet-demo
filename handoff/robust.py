import pandas as pd, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
h=pd.read_parquet("data/handoffs.parquet"); h["t"]=pd.to_datetime(h.t,utc=True); h["resp_t"]=pd.to_datetime(h.resp_t,utc=True)
L=np.log1p
h["f_repo"]=L(h.b_repo); h["f_tot"]=L(h.b_tot); h["f_hto"]=L(h.b_hto); h["f_hresp"]=L(h.b_hresp)
A=["f_repo","f_tot"]; B=A+["f_hto","f_hresp"]
def run(d,tr_end,te_end,ycol):
    tr=d[(d.t>=pd.Timestamp("2024-03-08",tz="UTC"))&(d.t<tr_end)]; te=d[(d.t>=tr_end)&(d.t<te_end)]
    r={}
    for n,f in (("answers",A),("plus_handoff",B)):
        m=LogisticRegression(max_iter=1000).fit(tr[f],tr[ycol]); r[n]=round(roc_auc_score(te[ycol],m.predict_proba(te[f])[:,1]),3)
    r["n_test"]=len(te); return r
out={}
for hrs in (24,72,168):
    h[f"y{hrs}"]=h.resp_t.notna()&((h.resp_t-h.t)<=pd.Timedelta(hours=hrs))
T=lambda s: pd.Timestamp(s,tz="UTC")
out["window_24h"]=run(h,T("2024-03-22"),T("2024-03-30"),"y24")
out["window_72h"]=run(h,T("2024-03-22"),T("2024-03-29"),"y72")
out["window_7d"]=run(h,T("2024-03-18"),T("2024-03-25"),"y168")
out["split_test_week3"]=run(h,T("2024-03-15"),T("2024-03-22"),"y72")
big=h.groupby("repo").issue.transform("size")>=50
out["repos_with_50plus_handoffs"]=run(h[big],T("2024-03-22"),T("2024-03-29"),"y72")
out["repos_small"]=run(h[~big],T("2024-03-22"),T("2024-03-29"),"y72")
# ranking: comments that hand off to >=2 new people at once, at least one responds
g=h[(h.t>=T("2024-03-22"))&(h.t<T("2024-03-29"))].copy()
tr=h[(h.t>=T("2024-03-08"))&(h.t<T("2024-03-22"))]
for n,f in (("answers",A),("plus_handoff",B)):
    g["p_"+n]=LogisticRegression(max_iter=1000).fit(tr[f],tr.y72).predict_proba(g[f])[:,1]
g["grp"]=g.issue+"|"+g.a+"|"+g.t.astype(str)
gg=g.groupby("grp").filter(lambda x: len(x)>=2 and x.y72.any())
res={"groups":gg.grp.nunique()}
for n in ("answers","plus_handoff"):
    res[n]=round(float(gg.loc[gg.groupby("grp")["p_"+n].idxmax()].y72.mean()),3)
res["random"]=round(float(gg.groupby("grp").y72.mean().mean()),3)
out["top1_among_multi_mentions"]=res
# export fleet-compatible weights (72h, all train)
fit={}
for n,f in (("answers",A),("plus_handoff",B)):
    m=LogisticRegression(max_iter=1000).fit(tr[f],tr.y72); fit[n]={"features":f,"coef":m.coef_[0].tolist(),"intercept":float(m.intercept_[0])}
json.dump(fit,open("data/github_models.json","w"),indent=1)
out["fleet_weights"]=fit
print(json.dumps(out,indent=1)); json.dump(out,open("data/robust_report.json","w"),indent=1)
