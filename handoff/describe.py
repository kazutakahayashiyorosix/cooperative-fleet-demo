import pandas as pd, numpy as np, json
df=pd.read_parquet("data/events.parquet",columns=["type","action","repo","key","actor"])
h=pd.read_parquet("data/handoffs.parquet")
out={"rows":len(df),"comments":int((df.type=="IssueComment").sum()),"issues_opened":int(((df.type=="Issues")&(df.action=="opened")).sum()),
 "issues_closed":int(((df.type=="Issues")&(df.action=="closed")).sum()),"repos":int(df.repo.nunique()),"issues_seen":int(df.key.nunique()),
 "actors":int(df.actor.nunique()),"handoffs":len(h),"handoff_repos":int(h.repo.nunique()),"handoff_targets":int(h.b.nunique()),
 "handoff_senders":int(h.a.nunique()),"resp72":round(float(h.resp72.mean()),3),
 "median_resp_hours":round(float(((pd.to_datetime(h.resp_t,utc=True)-pd.to_datetime(h.t,utc=True)).dt.total_seconds()/3600)[h.resp72].median()),1),
 "share_targets_no_prior_comments":round(float((h.b_tot==0).mean()),3)}
print(json.dumps(out,indent=1)); json.dump(out,open("data/describe.json","w"),indent=1)
