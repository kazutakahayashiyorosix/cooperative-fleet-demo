"""Figures, table and numbers.tex for the RA-L letter from results/ral.json and results/ral_extra.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RAL_OUT", os.path.join(HERE, "results"))
A = json.load(open(os.path.join(HERE, "results", "ral.json")))["summary"]
A.update(json.load(open(os.path.join(HERE, "results", "ral_extra.json")))["summary"])


def g(exp, pol, n, frac, chk="trusted"):
    return A[f"{exp}|{pol}|{n}|{frac}|{chk}"]


def per_req(d, k):
    return d[k][0] / d["requests"][0]


def unsafe_rate(d):
    return d["unsafe"][0] / max(d["unsafe"][0] + d["blocked"][0], 1e-9)


POL = [("broadcast", "Broadcast"), ("auction", "Auction"), ("random", "Random"),
       ("human", "Ledger (human)"), ("human_samecat", "Ledger, category only"), ("human_total", "Ledger, total only"),
       ("human_prox", "Ledger + proximity")]

# ---------- table: E1 at 0 and 0.3 unreliable
rows = []
for pol, lab in POL:
    r = [lab]
    for f in (0.0, 0.3, 0.5):
        d = g("E1", pol, 24, f)
        r += [f"{100*d['resolved_rate'][0]:.0f}", f"{per_req(d,'msgs'):.1f}"]
    d = g("E1", pol, 24, 0.0)
    r += [f"{d['mean_time_to_help_s'][0]:.1f}", f"{per_req(d,'wasted_travel_m'):.1f}"]
    rows.append(r)
with open(os.path.join(OUT, "table_e1.tex"), "w") as fh:
    for r in rows:
        fh.write(" & ".join(r) + r" \\" + "\n")

# ---------- numbers
m = {}
def put(k, v): m[k] = v
for f, t in ((0.0, "A"), (0.3, "B"), (0.5, "C")):
    for pol, key in (("broadcast", "Bc"), ("auction", "Au"), ("random", "Rw"), ("human", "Lg"), ("human_total", "Tot"), ("human_prox", "Px")):
        d = g("E1", pol, 24, f)
        put(f"res{key}{t}", f"{100*d['resolved_rate'][0]:.0f}")
        put(f"msg{key}{t}", f"{per_req(d,'msgs'):.1f}")
put("wasteBcA", f"{per_req(g('E1','broadcast',24,0.0),'wasted_travel_m'):.0f}")
put("timeBcA", f"{g('E1','broadcast',24,0.0)['mean_time_to_help_s'][0]:.1f}")
put("timeAuA", f"{g('E1','auction',24,0.0)['mean_time_to_help_s'][0]:.1f}")
put("timeLgA", f"{g('E1','human',24,0.0)['mean_time_to_help_s'][0]:.1f}")
put("msgRatioA", f"{per_req(g('E1','auction',24,0.0),'msgs')/per_req(g('E1','human',24,0.0),'msgs'):.0f}")
for n, t in ((8, "Eight"), (24, "TwentyFour"), (48, "FortyEight")):
    for pol, key in (("broadcast", "Bc"), ("auction", "Au"), ("human", "Lg"), ("human_prox", "Px")):
        d = g("E2", pol, n, 0.3)
        put(f"sc{key}{t}", f"{100*d['resolved_rate'][0]:.0f}")
        put(f"scmsg{key}{t}", f"{per_req(d,'msgs'):.1f}")
for f, t in ((0.2, "A"), (0.4, "B"), (0.6, "C")):
    for chk, key in (("nearest", "Ne"), ("trusted", "Tr")):
        put(f"cor{key}{t}", f"{100*unsafe_rate(g('E3','human',24,f,chk)):.0f}")
    put(f"corTw{t}", f"{100*unsafe_rate(g('E3','human',24,f,'two')):.0f}")
    for chk, key in (("nearest", "Ne"), ("trusted", "Tr"), ("two", "Tw")):
        put(f"ind{key}{t}", f"{100*unsafe_rate(g('E3b','human',24,f,chk)):.0f}")
put("indZero", f"{100*unsafe_rate(g('E3b','human',24,0.0,'nearest')):.1f}")
with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
    for k, v in sorted(m.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------- figure 1: scale, figure 2: peer check
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
ns = [8, 24, 48]
sty = {"broadcast": ("#7f7f7f", "o", "Broadcast"), "auction": ("#1f77b4", "s", "Auction"),
       "human": ("#d62728", "^", "Ledger (human)"), "human_prox": ("#ff7f0e", "v", "Ledger + proximity")}
for pol, (c, mk, lab) in sty.items():
    y = [100 * g("E2", pol, n, 0.3)["resolved_rate"][0] for n in ns]
    e = [100 * g("E2", pol, n, 0.3)["resolved_rate"][1] for n in ns]
    ax[0].errorbar(ns, y, e, color=c, marker=mk, ms=4, capsize=2, label=lab)
    y2 = [per_req(g("E2", pol, n, 0.3), "msgs") for n in ns]
    ax[1].plot(ns, y2, color=c, marker=mk, ms=4, label=lab)
ax[0].set_xlabel("fleet size"); ax[0].set_ylabel("requests resolved (%)"); ax[0].set_xticks(ns)
ax[1].set_xlabel("fleet size"); ax[1].set_ylabel("messages per request"); ax[1].set_yscale("log"); ax[1].set_xticks(ns)
ax[0].legend(fontsize=6.5, frameon=False, loc="lower left")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scale.pdf")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
fr = [0.0, 0.2, 0.4, 0.6]
for j, (exp, title) in enumerate((("E3", "compromised = unreliable"), ("E3b", "compromised independent of record"))):
    for chk, c, mk, lab in (("nearest", "#7f7f7f", "o", "nearest checker"), ("trusted", "#d62728", "^", "most trusted checker"),
                            ("two", "#1f77b4", "s", "two most trusted")):
        y = []
        for f in fr:
            key = "E3b" if (exp == "E3" and f == 0.0) else exp   # E3 at 0 equals E3b at 0
            try:
                y.append(100 * unsafe_rate(g(key, "human", 24, f, chk)))
            except KeyError:
                y.append(np.nan)
        ax[j].plot([100 * f for f in fr], y, color=c, marker=mk, ms=4, label=lab)
    ax[j].set_title(title, fontsize=8); ax[j].set_xlabel("compromised machines (%)")
    ax[j].spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("unsafe commands executed (%)"); ax[1].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_peer.pdf")); plt.close(fig)
print(len(m), "numbers")
