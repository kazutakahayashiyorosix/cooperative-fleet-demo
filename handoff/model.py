import pandas as pd, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
h=pd.read_parquet("data/handoffs.parquet")
t0=pd.Timestamp("2024-03-08",tz="UTC"); t1=pd.Timestamp("2024-03-22",tz="UTC"); t2=pd.Timestamp("2024-03-29",tz="UTC")
h["t"]=pd.to_datetime(h.t,utc=True)
h=h[(h.t>=t0)&(h.t<t2)].copy()
L=np.log1p
h["f_repo"]=L(h.b_repo); h["f_tot"]=L(h.b_tot); h["f_hto"]=L(h.b_hto); h["f_hresp"]=L(h.b_hresp)
h["f_rate"]=(h.b_hresp+1)/(h.b_hto+2)
tr=h[h.t<t1]; te=h[h.t>=t1]
sets={"answers only (like Stack Exchange)":["f_repo","f_tot"],
      "handoff history only":["f_hto","f_hresp","f_rate"],
      "answers + handoff history":["f_repo","f_tot","f_hto","f_hresp","f_rate"]}
out={"n_train":len(tr),"n_test":len(te),"base_rate_test":float(te.resp72.mean())}
for name,f in sets.items():
    m=LogisticRegression(max_iter=1000).fit(tr[f],tr.resp72)
    p=m.predict_proba(te[f])[:,1]; out[name]={"auc":round(roc_auc_score(te.resp72,p),3),"coef":dict(zip(f,np.round(m.coef_[0],3)))}
# bootstrap CI of AUC difference
rng=np.random.default_rng(0)
fa=sets["answers only (like Stack Exchange)"]; fb=sets["answers + handoff history"]
pa=LogisticRegression(max_iter=1000).fit(tr[fa],tr.resp72).predict_proba(te[fa])[:,1]
pb=LogisticRegression(max_iter=1000).fit(tr[fb],tr.resp72).predict_proba(te[fb])[:,1]
y=te.resp72.values; d=[]
for _ in range(300):
    ix=rng.integers(0,len(y),len(y)); d.append(roc_auc_score(y[ix],pb[ix])-roc_auc_score(y[ix],pa[ix]))
out["auc_gain_CI95"]=[round(float(np.percentile(d,2.5)),3),round(float(np.percentile(d,97.5)),3)]
# subgroup: targets with prior handoffs
sub=te.b_hto>0; out["test_share_with_prior_handoffs"]=round(float(sub.mean()),3)
out["auc_on_prior_handoff_targets"]={"answers":round(roc_auc_score(y[sub],pa[sub]),3),"plus_handoff":round(roc_auc_score(y[sub],pb[sub]),3)}
# simple descriptive: response rate by prior handoff record
te2=h[h.b_hto>=3]; out["resp_rate_by_prior_rate"]=te2.groupby(pd.cut(te2.f_rate,[0,.2,.4,.6,1.01])).resp72.agg(["mean","size"]).round(3).reset_index().astype(str).values.tolist()
print(json.dumps(out,indent=1,default=str))
json.dump(out,open("data/model_report.json","w"),indent=1,default=str)
