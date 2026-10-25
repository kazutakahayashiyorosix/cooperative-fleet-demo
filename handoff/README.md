# Handoff records (GitHub @-mention handoffs)

Code for "Who Answers When You Pass It On? Handoff Records Predict Help Beyond Help Records, from Issue Trackers to Robot Fleets".

1. `run.sh` (uses `hour.sh`, `extract.py`): downloads GH Archive for March 2024 and keeps issue events and issue comments with @-mentions (about 160 MB).
2. `load.py` → `analyze.py`: builds the handoff table causally (a comment mentioning someone not yet in the issue; uptake = that person comments within 72 h).
3. `model.py`, `robust.py`, `outcome.py`, `describe.py`: results in the `*.json` files here.
4. Robot fleet: `ral/run_ral.py handoff` uses the weights in `github_models.json` (policy `learned_handoff` in `ral/fleet_ext.py`).

No raw GitHub data are stored in this repository.
