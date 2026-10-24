"""
Extended version of fleet_sim.py for the RA-L experiments: configurable fleet size,
unreliable agents, a contract-net (auction) baseline, and compromised checkers.

MuJoCo simulation of a mixed fleet that routes help requests and checks each other.

Fleet: 2 drones (scout), 4 ground robots (2 lift, 2 fetch), 2 cars (deliver).
All bodies are physical: slide joints driven by velocity actuators, drones with
gravity compensation, payload boxes with free joints that fall and rest on the floor.

A request appears at a random spot and reaches the nearest machine. A machine that
can do it accepts and travels there; otherwise it passes the request on according
to the routing policy:
  broadcast : send to everyone; every able, idle machine sets off; first to arrive
              does the job, the others turn back (wasted travel)
  random    : pass to one machine at random
  learned   : pass to the machine the YOROSIX-trained help model scores highest
              (log1p of past help in this category, past help overall, past
              forwards, hop depth, "knows someone" flag). Ties go to the nearest.
A public ledger records who did what; it is the only thing machines know about
each other.

Peer check (CHALLENGE): now and then one machine receives an unsafe command (drive
out of the work area). With the peer check on, the most trusted nearby teammate
challenges it and the machine stops and returns home. With it off, the machine
leaves the area.

The routing model is trained on SYNTHETIC data; everything here is simulated.
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass, field
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = ["scout", "lift", "fetch", "deliver"]
COLORS = {"scout": (0.62, 0.70, 0.82), "lift": (1.0, 0.42, 0.29),
          "fetch": (0.18, 0.77, 0.71), "deliver": (0.95, 0.76, 0.31)}
FLEET = [  # name, skill, kind, home (x, y)
    ("D0", "scout", "drone", (-6.0, 5.5)), ("D1", "scout", "drone", (6.0, -5.5)),
    ("R0", "lift", "robot", (-3.5, -6.5)), ("R1", "lift", "robot", (4.5, 6.0)),
    ("R2", "fetch", "robot", (-7.0, -1.0)), ("R3", "fetch", "robot", (7.0, 1.5)),
    ("C0", "deliver", "car", (0.0, 7.5)), ("C1", "deliver", "car", (0.5, -7.5)),
]
VMAX = {"drone": 3.0, "robot": 1.6, "car": 3.5}
HEIGHT = {"drone": 3.2, "robot": 0.0, "car": 0.0}
DEPOT = np.array([0.0, 0.0])          # where lift / fetch jobs bring the payload
DROP = np.array([8.5, 8.5])           # where cars deliver
AREA_R = 10.0                         # work area radius; leaving it is unsafe
HOP_LATENCY = 0.4                     # seconds per message hop
N_PAYLOAD = 8
PEER_RANGE = 9.0                      # a teammate must be this close to challenge


KIND = {"scout": "drone", "lift": "robot", "fetch": "robot", "deliver": "car"}


def make_fleet(per_skill=2, seed=0):
    """per_skill agents of each skill, homes spread around the work area."""
    if per_skill == 2 and seed is None:
        return FLEET
    rng = np.random.default_rng(10_000 + seed)
    fleet, k = [], 0
    n = per_skill * len(SKILLS)
    angles = rng.permutation(np.linspace(0, 2 * math.pi, n, endpoint=False))
    for s_i, skill in enumerate(SKILLS):
        for j in range(per_skill):
            ang = angles[k]; r = rng.uniform(5.0, 8.0)
            fleet.append((f"{skill[0].upper()}{j}", skill, KIND[skill], (r * math.cos(ang), r * math.sin(ang))))
            k += 1
    return fleet


def build_xml(FLEET=FLEET):
    a = []
    for i, (name, skill, kind, (x, y)) in enumerate(FLEET):
        r, g, b = COLORS[skill]
        rgba = f"{r} {g} {b} 1"
        joints = (f'<joint name="{name}_x" type="slide" axis="1 0 0" damping="1"/>'
                  f'<joint name="{name}_y" type="slide" axis="0 1 0" damping="1"/>')
        if kind == "drone":
            joints += f'<joint name="{name}_z" type="slide" axis="0 0 1" damping="1"/>'
            geoms = (f'<geom type="box" size=".32 .32 .08" rgba=".14 .19 .23 1" mass="1.5"/>'
                     f'<geom type="box" size=".7 .04 .03" euler="0 0 45" rgba=".2 .26 .3 1" mass=".1"/>'
                     f'<geom type="box" size=".7 .04 .03" euler="0 0 -45" rgba=".2 .26 .3 1" mass=".1"/>'
                     + "".join(f'<geom type="cylinder" size=".26 .015" pos="{sx*.5} {sy*.5} .08" rgba="{rgba}" mass=".05"/>'
                               for sx in (-1, 1) for sy in (-1, 1)))
            body = f'<body name="{name}" pos="{x} {y} {HEIGHT[kind]}" gravcomp="1">{joints}{geoms}</body>'
        elif kind == "car":
            geoms = (f'<geom type="box" size=".8 .45 .22" pos="0 0 .42" rgba=".14 .19 .23 1" mass="8"/>'
                     f'<geom type="box" size=".42 .38 .18" pos="-.1 0 .8" rgba=".08 .11 .13 1" mass="1"/>'
                     f'<geom type="box" size=".81 .46 .04" pos="0 0 .56" rgba="{rgba}" mass=".2"/>'
                     + "".join(f'<geom type="cylinder" size=".2 .07" pos="{sx*.52} {sy*.47} .2" euler="90 0 0" rgba=".05 .07 .08 1" mass=".3"/>'
                               for sx in (-1, 1) for sy in (-1, 1)))
            body = f'<body name="{name}" pos="{x} {y} 0">{joints}{geoms}</body>'
        else:
            geoms = (f'<geom type="cylinder" size=".55 .45" pos="0 0 .45" rgba=".14 .19 .23 1" mass="6"/>'
                     f'<geom type="cylinder" size=".6 .05" pos="0 0 .92" rgba="{rgba}" mass=".2"/>'
                     f'<geom type="sphere" size=".26" pos="0 0 1.12" rgba=".06 .09 .11 1" mass=".3"/>')
            body = f'<body name="{name}" pos="{x} {y} 0">{joints}{geoms}</body>'
        a.append(body)
    payloads = "".join(
        f'<body name="P{k}" pos="{-11 + k*0.8} -11 .25"><freejoint/>'
        f'<geom type="box" size=".22 .22 .22" rgba=".85 .78 .62 1" mass=".5" contype="2" conaffinity="2"/></body>'
        for k in range(N_PAYLOAD))
    acts = []
    for name, skill, kind, _ in FLEET:
        for ax in ("x", "y") + (("z",) if kind == "drone" else ()):
            acts.append(f'<velocity joint="{name}_{ax}" kv="40" ctrlrange="-5 5"/>')
    ring = "".join(
        f'<geom type="box" size=".06 {AREA_R*math.sin(math.pi/24):.3f} .03" '
        f'pos="{AREA_R*math.cos(t):.3f} {AREA_R*math.sin(t):.3f} .03" euler="0 0 {math.degrees(t):.2f}" '
        f'rgba=".55 .62 .66 .6" contype="0" conaffinity="0"/>'
        for t in np.linspace(0, 2*math.pi, 24, endpoint=False))
    return f"""
<mujoco model="yorosix_fleet">
  <option timestep="0.01"/>
  <visual><headlight ambient=".35 .35 .35" diffuse=".5 .5 .5"/><global offwidth="1280" offheight="720"/>
          <quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".08 .11 .13" rgb2=".10 .14 .17" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance=".05"/>
  </asset>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="0 0 18" dir="0 0 -1" diffuse=".7 .7 .7" castshadow="true"/>
    <geom name="floor" type="plane" size="14 14 .1" material="grid" contype="2" conaffinity="2"/>
    <geom name="depot" type="cylinder" size=".9 .01" pos="{DEPOT[0]} {DEPOT[1]} .01" rgba=".3 .9 .6 .35"/>
    <geom name="drop" type="cylinder" size=".9 .01" pos="{DROP[0]} {DROP[1]} .01" rgba=".95 .76 .31 .35"/>
    {ring}
    {''.join(a)}
    {payloads}
  </worldbody>
  <actuator>{''.join(acts)}</actuator>
</mujoco>"""


# ------------------------------------------------------------------ routing model
class HelpModel:
    """Logistic help model. Reads any subset of the named features below, so both the
    original pipeline model (SYNTHETIC) and the Stack Exchange model (real human
    help-giving, se_help_model.py) plug in unchanged. Pick the file with the
    HELP_MODEL environment variable or the `model_path` argument."""
    NAMES = ["u_prior_acks_same_cat", "u_prior_acks", "u_prior_fwds", "depth", "arr_know_someone"]

    def __init__(self, path=None):
        path = path or os.environ.get("HELP_MODEL") or os.path.join(HERE, "help_model.json")
        if not os.path.isabs(path):
            path = os.path.join(HERE, path)
        m = json.load(open(path))
        feats = m.get("features", self.NAMES)
        self.idx = [self.NAMES.index(f) for f in feats]
        self.c, self.b = np.array(m["coef"]), m["intercept"]
        self.source = m.get("source", "")

    def score(self, same_cat, total, fwds, depth, know=1.0):
        full = [math.log1p(same_cat), math.log1p(total), math.log1p(fwds), depth, know]
        x = np.array([full[i] for i in self.idx])
        return 1 / (1 + math.exp(-(self.b + x @ self.c)))


@dataclass
class Ledger:
    acks: dict = field(default_factory=dict)
    total: dict = field(default_factory=dict)
    fwds: dict = field(default_factory=dict)

    def ack(self, i, cat):
        self.acks[(i, cat)] = self.acks.get((i, cat), 0) + 1
        self.total[i] = self.total.get(i, 0) + 1


@dataclass
class Agent:
    i: int; name: str; skill: str; kind: str; home: np.ndarray
    task: object = None              # job being done
    goal: np.ndarray = None
    rogue: float = 0.0               # >0 while following an unsafe command
    challenged: float = 0.0          # >0 while showing a challenge
    odo: float = 0.0
    leg: float = 0.0                 # distance driven on the current job
    wasted: float = 0.0              # distance driven toward jobs someone else took


@dataclass
class Job:
    jid: int; cat: str; loc: np.ndarray; t0: float
    payload: int = -1
    taker: int = -1                  # agent that does it
    racers: set = field(default_factory=set)   # agents heading there (broadcast)
    stage: str = "open"              # open -> going -> carrying -> done / dropped
    t_help: float = None
    hover: float = 0.0


class FleetSim:
    def __init__(self, policy="learned", peer_check=True, seed=0, request_every=3.0,
                 unsafe_every=12.0, model_path=None, per_skill=2, frac_unreliable=0.0,
                 p_accept_unreliable=0.2, checker="trusted", frac_compromised=None, prox_scale=5.0):
        fleet = make_fleet(per_skill, seed)
        self.fleet = fleet
        self.model = mujoco.MjModel.from_xml_string(build_xml(fleet))
        self.checker = checker
        self.data = mujoco.MjData(self.model)
        self.policy, self.peer_check = policy, peer_check
        self.rng = np.random.default_rng(seed)
        self.help = HelpModel(model_path)
        self.ledger = Ledger()
        self.request_every, self.unsafe_every = request_every, unsafe_every
        self.agents = [Agent(i, n, s, k, np.array(h, float)) for i, (n, s, k, h) in enumerate(fleet)]
        # unreliable agents accept a job they could do only with probability p_accept_unreliable;
        # they are also the compromised ones in the peer-check experiment (never challenge)
        n_bad = int(round(frac_unreliable * len(self.agents)))
        bad = set(self.rng.choice(len(self.agents), n_bad, replace=False).tolist()) if n_bad else set()
        self.p_acc = {a.i: (p_accept_unreliable if a.i in bad else 1.0) for a in self.agents}
        self.bad = bad
        # compromised checkers: by default the unreliable agents; or an independent random set
        if frac_compromised is None:
            self.compromised = set(bad)
        else:
            n_c = int(round(frac_compromised * len(self.agents)))
            rng_c = np.random.default_rng(777 + seed)
            self.compromised = set(rng_c.choice(len(self.agents), n_c, replace=False).tolist()) if n_c else set()
        self.prox_scale = prox_scale
        for a in self.agents:                                  # warm-up: past help on record, by reliability
            for _ in range(int(self.rng.binomial(10, self.p_acc[a.i]))):
                self.ledger.ack(a.i, a.skill)
        self.jobs, self.events, self.packets, self.ticker = [], [], [], []
        self.next_req, self.next_unsafe = 1.0, 6.0
        self.payload_free = list(range(N_PAYLOAD))
        self.stats = dict(requests=0, resolved=0, dropped=0, msgs=0, effective=0, sends=0,
                          blocked=0, unsafe=0)
        self.help_times = []
        self.qadr = {}
        for a in self.agents:
            for ax in ("x", "y", "z"):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{a.name}_{ax}")
                if jid >= 0:
                    self.qadr[(a.i, ax)] = (self.model.jnt_qposadr[jid], self.model.jnt_dofadr[jid])
        pj = [self.model.body_jntadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"P{k}")]
              for k in range(N_PAYLOAD)]
        self.pq = [int(self.model.jnt_qposadr[j]) for j in pj]
        self.pd = [int(self.model.jnt_dofadr[j]) for j in pj]
        self.bid = {a.i: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, a.name) for a in self.agents}
        self.ctrl_idx = {}
        k = 0
        for a in self.agents:
            for ax in ("x", "y") + (("z",) if a.kind == "drone" else ()):
                self.ctrl_idx[(a.i, ax)] = k; k += 1
        self.stats.update(bids=0)
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------- state helpers
    @property
    def t(self):
        return self.data.time

    def pos(self, a):
        return self.data.xpos[self.bid[a.i]].copy()

    def xy(self, a):
        return self.pos(a)[:2]

    def log(self, msg):
        self.ticker.insert(0, (self.t, msg)); del self.ticker[5:]

    def send(self, frm, to, kind, cb):
        """A message: arrives after HOP_LATENCY and is drawn while in flight."""
        self.stats["msgs"] += 1
        self.packets.append(dict(frm=frm, to=to, kind=kind, t0=self.t, t1=self.t + HOP_LATENCY))
        self.events.append((self.t + HOP_LATENCY, cb))

    # -------------------------------------------------------------- requests
    def new_request(self):
        cat = SKILLS[int(self.rng.integers(4))]
        ang, rad = self.rng.uniform(0, 2*math.pi), self.rng.uniform(1.5, 8.0)
        loc = np.array([rad*math.cos(ang), rad*math.sin(ang)])
        job = Job(len(self.jobs), cat, loc, self.t)
        if cat != "scout" and self.payload_free:
            job.payload = self.payload_free.pop(0)
            q = self.pq[job.payload]
            self.data.qpos[q:q+7] = [loc[0], loc[1], 0.6, 1, 0, 0, 0]
            self.data.qvel[self.pd[job.payload]:self.pd[job.payload]+6] = 0
        self.jobs.append(job); self.stats["requests"] += 1
        entry = min(self.agents, key=lambda a: np.linalg.norm(self.xy(a) - loc))
        self.log(f"Q{job.jid:03d} needs {cat} -> nearest {entry.name}")
        self.send(None, entry, "req", lambda: self.arrive(job, entry, 0, entry_hop=True))

    def able(self, a, job):
        return a.skill == job.cat and a.task is None and a.rogue <= 0

    def willing(self, a, job):
        """Able and, for an unreliable agent, chooses to accept this time."""
        return self.able(a, job) and (self.p_acc[a.i] >= 1.0 or self.rng.random() < self.p_acc[a.i])

    def arrive(self, job, a, depth, entry_hop=False, last_sender=None):
        if job.stage != "open":
            return
        if self.willing(a, job):
            if last_sender is not None:
                self.stats["effective"] += 1
            self.take(job, a); return
        if self.policy == "auction" and entry_hop:
            # contract net: call for proposals to all, willing agents bid, award to the nearest bidder
            others = [o for o in self.agents if o is not a]
            for o in others:
                self.stats["sends"] += 1; self.stats["msgs"] += 1
            bidders = [o for o in others if self.willing(o, job)]
            self.stats["bids"] += len(bidders); self.stats["msgs"] += len(bidders)
            if not bidders:
                self.events.append((self.t + HOP_LATENCY * 2, lambda: self.drop(job)))
                return
            win = min(bidders, key=lambda o: np.linalg.norm(self.xy(o) - job.loc))
            self.stats["effective"] += 1
            self.send(a, win, "award", lambda: self.award(job, win))
            return
        if self.policy == "broadcast" and entry_hop:
            others = [o for o in self.agents if o is not a]
            self.log(f"{a.name} cannot do {job.cat} -> broadcasts to all {len(others)}")
            for o in others:
                self.stats["sends"] += 1
                self.send(a, o, "fwd", (lambda o=o: self.broadcast_recv(job, o)))
            self.events.append((self.t + HOP_LATENCY * 2, lambda: self.check_dropped(job)))
            return
        cands = [o for o in self.agents if o is not a and o.rogue <= 0]
        if depth >= 4 or not cands:
            job.stage = "dropped"; self.stats["dropped"] += 1; self.release(job)
            self.log(f"{a.name} DROP Q{job.jid:03d}"); return
        if self.policy == "random":
            nxt = cands[int(self.rng.integers(len(cands)))]
        elif self.policy == "learned_prox":
            # ledger score weighted by closeness to the request
            sc = [(self.help.score(self.ledger.acks.get((o.i, job.cat), 0), self.ledger.total.get(o.i, 0),
                                   self.ledger.fwds.get(o.i, 0), depth + 1)
                   * math.exp(-np.linalg.norm(self.xy(o) - job.loc) / self.prox_scale), o) for o in cands]
            nxt = max(sc, key=lambda s: s[0])[1]
        else:
            sc = [(self.help.score(self.ledger.acks.get((o.i, job.cat), 0), self.ledger.total.get(o.i, 0),
                                   self.ledger.fwds.get(o.i, 0), depth + 1),
                   -np.linalg.norm(self.xy(o) - job.loc), o) for o in cands]
            nxt = max(sc, key=lambda s: (round(s[0], 6), s[1]))[2]
        self.stats["sends"] += 1
        self.ledger.fwds[a.i] = self.ledger.fwds.get(a.i, 0) + 1
        self.log(f"{a.name} cannot do {job.cat} -> passes to {nxt.name}")
        self.send(a, nxt, "fwd", lambda: self.arrive(job, nxt, depth + 1, last_sender=a))

    def drop(self, job):
        if job.stage == "open":
            job.stage = "dropped"; self.stats["dropped"] += 1; self.release(job)

    def award(self, job, o):
        # two extra hops (call, bid) before the award arrives
        if job.stage == "open" and self.able(o, job):
            self.events.append((self.t + HOP_LATENCY, lambda: self.take(job, o) if job.stage == "open" else None))
        else:
            self.drop(job)

    def broadcast_recv(self, job, o):
        if job.stage in ("open", "going") and self.willing(o, job):
            self.stats["effective"] += 1
            job.racers.add(o.i); o.task = job; o.goal = job.loc.copy()
            if job.stage == "open":
                job.stage = "going"
                self.log(f"{o.name} sets off for Q{job.jid:03d}")

    def check_dropped(self, job):
        if job.stage == "open":
            job.stage = "dropped"; self.stats["dropped"] += 1; self.release(job)

    def take(self, job, a):
        job.stage, job.taker = "going", a.i
        job.racers = {a.i}
        a.task, a.goal = job, job.loc.copy()
        self.log(f"{a.name} ACK - doing {job.cat}")

    def release(self, job):
        if job.payload >= 0:
            q = self.pq[job.payload]
            self.data.qpos[q:q+7] = [-11 + job.payload*0.8, -11, 0.25, 1, 0, 0, 0]
            self.payload_free.append(job.payload); job.payload = -1

    # -------------------------------------------------------------- peer check
    def unsafe_command(self):
        idle = [a for a in self.agents if a.task is None and a.rogue <= 0]
        if not idle:
            return
        r = idle[int(self.rng.integers(len(idle)))]
        d = self.xy(r) / (np.linalg.norm(self.xy(r)) + 1e-6)
        r.rogue, r.goal = 1.0, d * (AREA_R + 3.0)
        self.log(f"{r.name} receives an unsafe command: leave the work area")
        if self.peer_check:
            peers = [o for o in self.agents if o is not r and o.rogue <= 0
                     and np.linalg.norm(self.xy(o) - self.xy(r)) < PEER_RANGE]
            if not peers:
                self.log(f"no teammate within {PEER_RANGE:.0f} m of {r.name}"); return
            if self.checker == "nearest":
                chks = [min(peers, key=lambda o: np.linalg.norm(self.xy(o) - self.xy(r)))]
            elif self.checker == "two":   # the two most trusted teammates both check
                chks = sorted(peers, key=lambda o: (-self.ledger.total.get(o.i, 0), np.linalg.norm(self.xy(o) - self.xy(r))))[:2]
            else:   # trusted: most help on the public ledger, nearest on ties
                chks = [max(peers, key=lambda o: (self.ledger.total.get(o.i, 0), -np.linalg.norm(self.xy(o) - self.xy(r))))]
            for chk in chks:
                if chk.i in self.compromised:
                    self.log(f"{chk.name} is compromised and does not challenge"); continue
                self.send(chk, r, "chal", lambda chk=chk: self.challenge(chk, r))

    def challenge(self, chk, r):
        if r.rogue > 0:
            r.rogue, r.goal, r.challenged = 0.0, None, 2.0
            self.stats["blocked"] += 1
            self.log(f"{chk.name} CHALLENGE - stops {r.name}")

    # -------------------------------------------------------------- control loop
    def control(self):
        for a in self.agents:
            p = self.pos(a)
            job = a.task
            if job is not None:
                if job.stage == "dropped" or (job.stage in ("carrying", "done") and job.taker != a.i):
                    a.wasted += a.leg; a.task = None; a.goal = None
                elif job.stage == "going" and np.linalg.norm(p[:2] - job.loc) < 0.5:
                    if job.taker == -1 or job.taker == a.i:
                        job.taker = a.i; job.t_help = self.t - job.t0; self.help_times.append(job.t_help)
                        self.ledger.ack(a.i, job.cat); self.stats["resolved"] += 1
                        for o in self.agents:                       # losers of a broadcast race turn back
                            if o.task is job and o is not a:
                                o.wasted += o.leg; o.task, o.goal = None, None
                                self.log(f"{o.name} turns back ({o.leg:.1f} m wasted)")
                        if job.cat == "scout":
                            job.stage, job.hover = "carrying", 2.0
                        else:
                            job.stage = "carrying"
                            a.goal = (DROP if job.cat == "deliver" else DEPOT).copy()
                elif job.stage == "carrying" and job.taker == a.i:
                    if job.cat == "scout":
                        job.hover -= self.model.opt.timestep
                        if job.hover <= 0:
                            job.stage = "done"; a.task, a.goal = None, None
                    else:
                        q = self.pq[job.payload]
                        top = 2.3 if a.kind == "drone" else (1.0 if a.kind == "car" else 1.5)
                        self.data.qpos[q:q+3] = [p[0], p[1], top]
                        self.data.qvel[self.pd[job.payload]:self.pd[job.payload]+6] = 0
                        if np.linalg.norm(p[:2] - a.goal) < 0.6:
                            job.stage = "done"; self.release(job); a.task, a.goal = None, None
            # wasted travel: moving toward a job someone else took
            goal = a.goal if a.goal is not None else a.home
            dxy = goal - p[:2]
            dist = np.linalg.norm(dxy)
            v = dxy / (dist + 1e-9) * min(VMAX[a.kind], 1.5 * dist)
            self.data.ctrl[self.ctrl_idx[(a.i, "x")]] = v[0]
            self.data.ctrl[self.ctrl_idx[(a.i, "y")]] = v[1]
            if a.kind == "drone":
                z = self.data.qpos[self.qadr[(a.i, "z")][0]]       # offset from cruise height
                self.data.ctrl[self.ctrl_idx[(a.i, "z")]] = float(np.clip(-2.0 * z, -2, 2))
            if a.rogue > 0 and np.linalg.norm(p[:2]) > AREA_R:
                self.stats["unsafe"] += 1; a.rogue, a.goal = 0.0, None
                self.log(f"{a.name} left the work area (unsafe action went through)")
            a.challenged = max(0.0, a.challenged - self.model.opt.timestep)

    def step(self, n=1):
        for _ in range(n):
            if self.t >= self.next_req:
                self.new_request(); self.next_req += self.request_every
            if self.t >= self.next_unsafe:
                self.unsafe_command(); self.next_unsafe += self.unsafe_every
            due = [e for e in self.events if e[0] <= self.t]
            self.events = [e for e in self.events if e[0] > self.t]
            for _, cb in sorted(due, key=lambda e: e[0]):
                cb()
            self.packets = [p for p in self.packets if p["t1"] > self.t]
            before = {a.i: self.xy(a) for a in self.agents}
            self.control()
            mujoco.mj_step(self.model, self.data)
            for a in self.agents:
                d = float(np.linalg.norm(self.xy(a) - before[a.i]))
                a.odo += d
                a.leg = a.leg + d if a.task is not None else 0.0

    def metrics(self):
        s = dict(self.stats)
        s["resolved_rate"] = s["resolved"] / max(1, s["requests"])
        s["mean_time_to_help_s"] = float(np.mean(self.help_times)) if self.help_times else None
        s["wasted_msgs_per_resolved"] = (s["sends"] - s["effective"]) / max(1, s["resolved"])
        s["distance_total_m"] = float(sum(a.odo for a in self.agents))
        s["wasted_travel_m"] = float(sum(a.wasted for a in self.agents))
        s["sim_seconds"] = float(self.t)
        return s
