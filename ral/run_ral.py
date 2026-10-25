"""
Experiments for the RA-L letter.

E1  routing: 24 agents (6 per skill), share of unreliable agents 0 / 0.3 / 0.5,
    policies broadcast, auction (contract net), random, and the ledger rule with weights
    fitted to real human help-giving (both features, and each feature alone).
E2  scale: 8 / 24 / 48 agents at 30% unreliable, broadcast vs auction vs human ledger.
E3  peer check: 24 agents, share of compromised agents 0..0.6 (they never challenge),
    checker chosen as the nearest teammate or the most trusted on the ledger.

Each run: 300 simulated seconds; 10 seeds. Writes results/ral.json.
"""
import json, os, sys, itertools
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fleet_ext import FleetSim

ROOT = os.path.dirname(HERE)
MODELS = {"human": os.path.join(ROOT, "help_model_human.json"),
          "human_samecat": os.path.join(HERE, "help_model_human_samecat.json"),
          "human_total": os.path.join(HERE, "help_model_human_total.json")}
SIM_S = float(os.environ.get("SIM_SECONDS", 300))
SEEDS = range(int(os.environ.get("SEEDS", 10)))
KEYS = ["requests", "resolved_rate", "mean_time_to_help_s", "sends", "bids", "wasted_travel_m",
        "blocked", "unsafe", "msgs"]


def run(job):
    exp, pol, per_skill, frac, checker, seed = job[:6]
    comp = job[6] if len(job) > 6 else None
    policy = "learned_prox" if pol in ("human_prox", "human_avail_prox") else ("learned" if pol.startswith("human") else pol)
    s = FleetSim(policy, peer_check=True, seed=seed, model_path=MODELS.get(pol, MODELS["human"]),
                 per_skill=per_skill, frac_unreliable=frac, checker=checker,
                 request_every=3.0 * 2 / per_skill, unsafe_every=(4.0 if exp.startswith("E3") else 12.0),
                 frac_compromised=comp, avail=pol.startswith("human_avail"))
    s.step(int(SIM_S / s.model.opt.timestep))
    m = s.metrics()
    return dict(exp=exp, policy=pol, n=per_skill * 4, frac=(frac if comp is None else comp), checker=checker, seed=seed,
                **{k: m.get(k) for k in KEYS})


def jobs_extra():
    J = []
    for frac, sd in itertools.product([0.0, 0.3, 0.5], SEEDS):
        J.append(("E1", "human_prox", 6, frac, "trusted", sd))
    for per, sd in itertools.product([2, 6, 12], SEEDS):
        J.append(("E2", "human_prox", per, 0.3, "trusted", sd))
    # E3b: compromised agents chosen independently of reliability (all agents reliable)
    for comp, chk, sd in itertools.product([0.0, 0.2, 0.4, 0.6], ["nearest", "trusted", "two"], SEEDS):
        J.append(("E3b", "human", 6, 0.0, chk, sd, comp))
    for comp, sd in itertools.product([0.2, 0.4, 0.6], SEEDS):
        J.append(("E3", "human", 6, comp, "two", sd))
    return J


def jobs_avail():
    J = []
    for frac, pol, sd in itertools.product([0.0, 0.3, 0.5], ["human_avail", "human_avail_prox"], SEEDS):
        J.append(("E1", pol, 6, frac, "trusted", sd))
    for per, pol, sd in itertools.product([2, 6, 12], ["human_avail", "human_avail_prox"], SEEDS):
        J.append(("E2", pol, per, 0.3, "trusted", sd))
    return J


def jobs():
    J = []
    for frac, pol, sd in itertools.product([0.0, 0.3, 0.5],
                                           ["broadcast", "auction", "random", "human", "human_samecat", "human_total"], SEEDS):
        J.append(("E1", pol, 6, frac, "trusted", sd))
    for per, pol, sd in itertools.product([2, 6, 12], ["broadcast", "auction", "human"], SEEDS):
        J.append(("E2", pol, per, 0.3, "trusted", sd))
    for frac, chk, sd in itertools.product([0.0, 0.2, 0.4, 0.6], ["nearest", "trusted"], SEEDS):
        J.append(("E3", "human", 6, frac, chk, sd))
    return J


def summarise(rows):
    out = {}
    for r in rows:
        key = f'{r["exp"]}|{r["policy"]}|{r["n"]}|{r["frac"]}|{r["checker"]}'
        out.setdefault(key, []).append(r)
    summ = {}
    for k, rs in out.items():
        d = {}
        for m in KEYS:
            v = np.array([np.nan if x[m] is None else float(x[m]) for x in rs])
            d[m] = [float(np.nanmean(v)), float(np.nanstd(v))]
        d["n_runs"] = len(rs)
        summ[k] = d
    return summ


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    extra = mode in ("extra", "avail")
    J = {"extra": jobs_extra, "avail": jobs_avail}.get(mode, jobs)()
    with Pool() as p:
        rows = p.map(run, J, chunksize=2)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", {"extra": "ral_extra.json", "avail": "ral_avail.json"}.get(mode, "ral.json"))
    json.dump(dict(sim_seconds=SIM_S, seeds=len(SEEDS), summary=summarise(rows), runs=rows), open(out, "w"), indent=1)
    print("done", len(rows))
