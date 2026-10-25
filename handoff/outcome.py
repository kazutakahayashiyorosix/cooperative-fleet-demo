import pandas as pd, numpy as np, json
h=pd.read_parquet("data/handoffs.parquet"); iss=pd.read_parquet("data/issues.parquet")
h["t"]=pd.to_datetime(h.t,utc=True); iss["open_t"]=pd.to_datetime(iss.open_t,utc=True); iss["close_t"]=pd.to_datetime(iss.close_t,utc=True)
# issues opened Mar 1-21, outcome closed within 7 days of first handoff (or of opening for no-handoff)
iss=iss[(iss.open_t<pd.Timestamp("2024-03-22",tz="UTC"))]
first=h[h.in_window].sort_values("t").groupby("issue").first()
d=iss.set_index("key").join(first[["t","resp72","b_hto","b_hresp","b_tot"]],how="inner")
d["repo"]=d.index.str.split("#").str[0]
d["closed7"]=(d.close_t.notna())&((d.close_t-d.t)<=pd.Timedelta(days=7))&(d.close_t>=d.t)
d=d[d.t<pd.Timestamp("2024-03-24",tz="UTC")]
out={"issues_with_handoff":len(d),"closed7_if_target_responded":round(d[d.resp72].closed7.mean(),3),
     "closed7_if_not":round(d[~d.resp72].closed7.mean(),3)}
# within-repo: repos with both types
g=d.groupby("repo").filter(lambda x: x.resp72.nunique()==2)
w=g.groupby(["repo","resp72"]).closed7.mean().unstack()
out["within_repo_n_repos"]=len(w); out["within_repo_mean_diff"]=round(float((w[True]-w[False]).mean()),3)
diffs=(w[True]-w[False]).values; rng=np.random.default_rng(0)
bs=[rng.choice(diffs,len(diffs)).mean() for _ in range(1000)]
out["within_repo_CI95"]=[round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)]
print(json.dumps(out,indent=1)); json.dump(out,open("data/outcome_report.json","w"),indent=1)
