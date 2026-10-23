"""
Train the fleet's routing model on REAL human help-giving behaviour from a
Stack Exchange data dump (CC BY-SA 4.0, https://archive.org/details/stackexchange).

Question the model answers: when someone needs help in category c, which
candidate is most likely to actually help? On Stack Exchange a "request" is a
question, its category is its tags, and "helping" is posting an answer
(optionally: an accepted answer).

For every question q (time order) we build candidate rows:
  - every user who answered q            -> label 1
  - k sampled users who had answered something before q but not q -> label 0
Features use only history strictly before q (no leakage):
  u_prior_acks_same_cat : answers this user gave earlier to questions sharing a tag with q
  u_prior_acks          : answers this user gave earlier in total
These are the same two quantities the robots read from their public ledger
("help given in this category", "help given overall"), so the coefficients
transfer directly to fleet_sim.HelpModel.

Usage:
  python se_help_model.py robotics.stackexchange.com.7z          # or a folder / Posts.xml
  python se_help_model.py --fixture                              # tiny synthetic test file
Writes help_model_human.json and out/se_model_report.json.
"""
from __future__ import annotations
import argparse, io, json, math, os, sys
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = ["u_prior_acks_same_cat", "u_prior_acks"]


# ------------------------------------------------------------------ reading the dump
def open_posts(path: str):
    """Return a binary file object for Posts.xml from a .7z archive, a folder or the file itself."""
    if path.endswith(".7z"):
        import py7zr, tempfile
        tmp = tempfile.mkdtemp(prefix="se_dump_")
        with py7zr.SevenZipFile(path, "r") as z:
            names = [n for n in z.getnames() if n.lower().endswith("posts.xml")]
            if not names:
                sys.exit("Posts.xml not found in archive")
            z.extract(path=tmp, targets=names[:1])
        return open(os.path.join(tmp, names[0]), "rb")
    if os.path.isdir(path):
        path = os.path.join(path, "Posts.xml")
    return open(path, "rb")


def parse_tags(s: str | None) -> tuple:
    if not s:
        return ()
    s = s.strip()
    if s.startswith("|"):                      # newer dumps: |tag1|tag2|
        return tuple(t for t in s.split("|") if t)
    return tuple(t for t in s.replace(">", "").split("<") if t)   # older: <tag1><tag2>


def load(path: str):
    questions, answers = {}, []
    for _, el in ET.iterparse(open_posts(path), events=("end",)):
        if el.tag != "row":
            continue
        a = el.attrib
        t = a.get("PostTypeId")
        if t == "1":
            questions[a["Id"]] = dict(t=a["CreationDate"], tags=parse_tags(a.get("Tags")),
                                      owner=a.get("OwnerUserId"), accepted=a.get("AcceptedAnswerId"))
        elif t == "2" and a.get("OwnerUserId"):
            answers.append(dict(id=a["Id"], q=a.get("ParentId"), t=a["CreationDate"], user=a["OwnerUserId"]))
        el.clear()
    return questions, answers


# ------------------------------------------------------------------ building decision rows
def build_rows(questions, answers, k_neg=20, seed=0, label="answer"):
    rng = np.random.default_rng(seed)
    by_q = defaultdict(list)
    for a in answers:
        if a["q"] in questions:
            by_q[a["q"]].append(a)
    accepted_ids = {q["accepted"] for q in questions.values() if q["accepted"]}
    hist_cat = defaultdict(int)       # (user, tag) -> answers before now
    hist_all = defaultdict(int)       # user -> answers before now
    seen_users: list = []; seen_set = set()
    rows = []
    order = sorted(questions.items(), key=lambda kv: kv[1]["t"])
    for qid, q in order:
        ans = by_q.get(qid, [])
        if label == "accepted":
            pos = {a["user"] for a in ans if a["id"] in accepted_ids}
        else:
            pos = {a["user"] for a in ans}
        pos.discard(q["owner"])
        if seen_users:
            pool = [u for u in rng.choice(seen_users, size=min(len(seen_users), k_neg * 3), replace=False)
                    if u not in pos and u != q["owner"]][:k_neg]
            for u, y in [(u, 1) for u in pos if u in seen_set] + [(u, 0) for u in pool]:
                same = max((hist_cat[(u, tg)] for tg in q["tags"]), default=0)
                rows.append((q["t"], qid, u, same, hist_all[u], y))
        for a in ans:                                   # history is updated after the question is scored
            u = a["user"]
            for tg in q["tags"]:
                hist_cat[(u, tg)] += 1
            hist_all[u] += 1
            if u not in seen_set:
                seen_set.add(u); seen_users.append(u)
    return rows


# ------------------------------------------------------------------ fit and evaluate
def fit(rows, test_frac=0.2):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rows = sorted(rows, key=lambda r: r[0])
    X = np.array([[math.log1p(r[3]), math.log1p(r[4])] for r in rows])
    y = np.array([r[5] for r in rows]); qid = np.array([r[1] for r in rows])
    cut = int(len(rows) * (1 - test_frac))
    m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[:cut], y[:cut])
    p = m.predict_proba(X[cut:])[:, 1]
    rep = {"rows": int(len(rows)), "positives": int(y.sum()), "test_rows": int(len(rows) - cut),
           "test_auc": float(roc_auc_score(y[cut:], p)),
           "test_auc_total_only": float(roc_auc_score(y[cut:], X[cut:, 1])),
           "test_auc_same_cat_only": float(roc_auc_score(y[cut:], X[cut:, 0]))}
    # routing accuracy: per test question, is the top-scored candidate one who actually helped?
    hit = {"model": [], "most_active": [], "random": []}
    rng = np.random.default_rng(1)
    te = slice(cut, None)
    for q in np.unique(qid[te]):
        idx = np.where(qid[te] == q)[0]
        yy = y[te][idx]
        if yy.sum() == 0 or len(idx) < 2:
            continue
        hit["model"].append(yy[np.argmax(p[idx])])
        hit["most_active"].append(yy[np.argmax(X[te][idx, 1])])
        hit["random"].append(yy[rng.integers(len(idx))])
    rep["top1_hit_rate"] = {k: float(np.mean(v)) for k, v in hit.items()}
    rep["questions_scored"] = len(hit["model"])
    model = {"features": FEATS, "coef": [float(c) for c in m.coef_[0]], "intercept": float(m.intercept_[0])}
    return model, rep


# ------------------------------------------------------------------ test fixture
def write_fixture(path, n_users=60, n_q=900, seed=7):
    """Small SYNTHETIC Posts.xml in Stack Exchange format, for testing the code path only."""
    rng = np.random.default_rng(seed)
    tags = ["ros", "slam", "kinematics", "arduino", "control", "sensors"]
    expert = {u: rng.choice(tags, 2, replace=False) for u in range(n_users)}
    act = rng.lognormal(0, 1, n_users)
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<posts>"]
    pid = 1
    for i in range(n_q):
        d = i * 330 // n_q
        t = f"2020-{1 + d // 28:02d}-{1 + d % 28:02d}T{i % 24:02d}:00:00.000"
        tg = tuple(rng.choice(tags, rng.integers(1, 3), replace=False))
        qid = pid; pid += 1
        out.append(f'<row Id="{qid}" PostTypeId="1" CreationDate="{t}" OwnerUserId="{1000 + i}" Tags="|{"|".join(tg)}|" />')
        w = act * np.array([3.0 if set(tg) & set(expert[u]) else 0.3 for u in range(n_users)])
        for u in rng.choice(n_users, rng.integers(0, 3), replace=False, p=w / w.sum()):
            out.append(f'<row Id="{pid}" PostTypeId="2" ParentId="{qid}" CreationDate="{t}" OwnerUserId="{u}" />'); pid += 1
    out.append("</posts>")
    open(path, "w").write("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--label", choices=["answer", "accepted"], default="answer")
    ap.add_argument("--out", default=os.path.join(HERE, "help_model_human.json"))
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    if a.fixture:
        a.dump = os.path.join(HERE, "out", "fixture_Posts.xml"); write_fixture(a.dump)
        a.out = os.path.join(HERE, "out", "help_model_fixture.json")
    if not a.dump:
        ap.error("give a Stack Exchange dump (.7z, folder or Posts.xml) or --fixture")
    qs, ans = load(a.dump)
    rows = build_rows(qs, ans, label=a.label)
    model, rep = fit(rows)
    src = os.path.basename(a.dump)
    model["source"] = ("SYNTHETIC test fixture" if a.fixture else
                       f"Stack Exchange data dump {src} (real human help-giving, CC BY-SA 4.0), label={a.label}")
    rep.update(source=model["source"], questions=len(qs), answers=len(ans), coef=dict(zip(FEATS, model["coef"])))
    json.dump(model, open(a.out, "w"), indent=1)
    json.dump(rep, open(os.path.join(HERE, "out", "se_model_report.json" if not a.fixture else "fixture_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
