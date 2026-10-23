# Cooperative fleet demo: machines that route help and check each other

Part of [YOROSIX](https://yorosix.com), a project that records how people ask for help, reply, rate replies
and pass requests on, and turns those decisions into training data for cooperative AI.

**What is real and what is simulated.** The machines, the jobs and the physics are simulated (MuJoCo).
The routing rule can be trained on real human help-giving (a public Stack Exchange dump, see below);
the default `help_model.json` comes from the YOROSIX pipeline on SYNTHETIC data. Status: the Stack Exchange
training code is tested on a synthetic fixture; results on a real dump will be added here.

The machines are simulated with physics: 2 drones (scout), 4 ground robots (2 lift, 2 fetch) and 2 cars (deliver).
A request (a job in one category) appears in the work area and first reaches the nearest machine.
If that machine can't do it, it passes the request on.

- **broadcast**: send to everyone. Every able, idle machine sets off; the first to arrive does the job and the rest turn back.
- **random**: pass to one machine chosen at random.
- **learned**: pass to the machine ranked highest by the help model trained in the YOROSIX pipeline (SYNTHETIC data).
  - Model features: past help in the same category, total help, forward count, hop depth.
  - The only information machines have about each other is the public ledger (who did what).

**Peer check (CHALLENGE)**: an unsafe command ("leave the work area") is sent to one machine at a time.
With the check on, the most trusted teammate within 9 m challenges it and stops it.

## Setup

```bash
pip install -r requirements.txt
# ffmpeg is needed only for video (brew install ffmpeg)
```

## Run

```bash
mjpython view_live.py              # live 3D on a Mac (learned + peer check)
mjpython view_live.py broadcast    # compare
mjpython view_live.py learned off  # peer check off
python run_benchmark.py            # 10 seeds x 300 s -> out/benchmark.json, .png
python make_video.py routing       # side by side: broadcast vs learned -> out/fleet_routing.mp4
python make_video.py peer          # side by side: peer check off vs on -> out/fleet_peercheck.mp4
```

On headless Linux, prefix commands with `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.

## Results (10 seeds × 300 s, 100 requests each)

| | broadcast | random | learned |
|---|---|---|---|
| Resolved | 90% | 69% | **97%** |
| Time to help | **3.5 s** | 4.6 s | 4.3 s |
| Messages sent | 576 | 235 | **111** |
| Wasted travel | 347 m | 0 | **0** |

Peer check (learned, 25 unsafe commands): **0.5** executed with the check on, **25** with it off.

**Caveats:**
- Broadcast is faster to first arrival because the nearest machine races there. The cost is 5× the messages, wasted travel, and machines that are busy when the next job comes in.
- The help model is trained on synthetic data. No real-data performance is claimed here.
- Machine dynamics are simplified: slide joints plus velocity control. This is not flight control or vehicle dynamics.

## Routing model trained on real human help-giving (Stack Exchange)

`se_help_model.py` trains the same routing rule on **real people helping each other**: a Stack Exchange data dump
(CC BY-SA 4.0, https://archive.org/details/stackexchange). A question is a request, its tags are the category,
and posting an answer is helping. For each question the model sees everyone who answered it and a sample of
users who had answered before but not this time. Features use only history before the question:

- help given earlier in the same category (answers to questions sharing a tag)
- help given earlier in total

These are exactly what the machines read from their public ledger, so the coefficients drop straight into the fleet.

```bash
pip install py7zr scikit-learn
python se_help_model.py robotics.stackexchange.com.7z      # -> help_model_human.json, out/se_model_report.json
python run_benchmark.py                                    # adds a "human" column when help_model_human.json exists
HELP_MODEL=help_model_human.json python make_video.py routing
python se_help_model.py --fixture                          # code-path test on a small SYNTHETIC file
```

What is real and what is not: the human behaviour the model learns from is real; the machines, the jobs and
the physics are simulated. The report gives held-out AUC (time split) and how often the top-ranked candidate
was someone who actually helped, against "most active overall" and random baselines.

### Path to NVIDIA Isaac Lab / Newton

The routing layer is independent of the physics: `FleetSim` only needs positions, a move command and an
"arrived" event. The MJCF world can be loaded by Newton (which builds on MuJoCo Warp) and run inside Isaac Lab;
the routing model and ledger are plain Python/NumPy and move over unchanged. This port has not been run yet
(it needs an NVIDIA GPU).

## Files

- `fleet_sim.py`: model (MJCF generation), routing, peer check, metrics
- `help_model.json`: help-model coefficients (from the YOROSIX pipeline, SYNTHETIC)
- `se_help_model.py`: trains `help_model_human.json` from a Stack Exchange dump (real human data)
- `run_benchmark.py` / `make_video.py` / `view_live.py`
