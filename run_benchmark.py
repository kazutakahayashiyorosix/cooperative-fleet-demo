"""Benchmark: routing policy x peer check, several seeds. Writes out/benchmark.json and out/benchmark.png."""
import json, os, sys
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fleet_sim import FleetSim

SIM_SECONDS = float(os.environ.get("SIM_SECONDS", 300))
SEEDS = range(int(os.environ.get("SEEDS", 10)))
KEYS = ["resolved_rate", "mean_time_to_help_s", "sends", "wasted_msgs_per_resolved",
        "wasted_travel_m", "distance_total_m", "blocked", "unsafe", "requests"]


HUMAN_MODEL = os.environ.get("HUMAN_MODEL", os.path.join(HERE, "help_model_human.json"))
POLS = ["broadcast", "random", "learned"] + (["human"] if os.path.exists(HUMAN_MODEL) else [])


def run(args):
    policy, peer, seed = args
    if policy == "human":            # same routing rule, model trained on real Stack Exchange answers
        s = FleetSim("learned", peer_check=peer, seed=seed, model_path=HUMAN_MODEL)
    else:
        s = FleetSim(policy, peer_check=peer, seed=seed)
    s.step(int(SIM_SECONDS / s.model.opt.timestep))
    m = s.metrics()
    return dict(policy=policy, peer_check=peer, seed=seed, **{k: m[k] for k in KEYS})


def main():
    jobs = [(p, True, s) for p in POLS for s in SEEDS]
    jobs += [("learned", False, s) for s in SEEDS]
    with Pool() as pool:
        rows = pool.map(run, jobs)
    out = os.path.join(HERE, "out"); os.makedirs(out, exist_ok=True)
    summ = {}
    for p in POLS:
        r = [x for x in rows if x["policy"] == p and x["peer_check"]]
        summ[p] = {k: [float(np.mean([x[k] for x in r])), float(np.std([x[k] for x in r]))] for k in KEYS}
    for peer in (True, False):
        r = [x for x in rows if x["policy"] == "learned" and x["peer_check"] == peer]
        summ[f"peer_check_{'on' if peer else 'off'}"] = {k: [float(np.mean([x[k] for x in r])), float(np.std([x[k] for x in r]))]
                                                       for k in ("blocked", "unsafe")}
    json.dump(dict(sim_seconds=SIM_SECONDS, seeds=len(SEEDS), summary=summ, runs=rows),
              open(os.path.join(out, "benchmark.json"), "w"), indent=1)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pols = POLS; col = ["#9aa7b4", "#c9b48a", "#2ec4b6", "#3a6fd8"][:len(POLS)]
    panels = [("resolved_rate", "Requests resolved", 100, "%"), ("mean_time_to_help_s", "Time to help", 1, "s"),
              ("sends", "Messages sent", 1, ""), ("wasted_travel_m", "Wasted travel", 1, "m")]
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
    for k, (key, title, mul, unit) in enumerate(panels):
        m = [summ[p][key][0] * mul for p in pols]; e = [summ[p][key][1] * mul for p in pols]
        ax[k].bar(pols, m, yerr=e, color=col, capsize=4)
        ax[k].set_title(f"{title}" + (f" ({unit})" if unit else ""), fontsize=11)
        for i, v in enumerate(m):
            ax[k].text(i, v + e[i], f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    on, off = summ["peer_check_on"], summ["peer_check_off"]
    ax[4].bar(["check on", "check off"], [on["unsafe"][0], off["unsafe"][0]], color=["#2ec4b6", "#ff6b4a"])
    ax[4].set_title("Unsafe actions executed", fontsize=11)
    for i, v in enumerate([on["unsafe"][0], off["unsafe"][0]]):
        ax[4].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    note = "learned = SYNTHETIC pipeline model" + ("; human = model trained on real Stack Exchange answers" if "human" in POLS else "")
    fig.suptitle(f"MuJoCo fleet, {len(SEEDS)} seeds x {SIM_SECONDS:.0f} s each  ({note})", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "benchmark.png"), dpi=130)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
