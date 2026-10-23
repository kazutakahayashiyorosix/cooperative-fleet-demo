"""Interactive MuJoCo viewer.

  python view_live.py                 # learned routing, peer check on
  python view_live.py broadcast       # broadcast / random / learned
  python view_live.py learned off     # peer check off

On macOS, use mjpython (installed with the mujoco package):
  mjpython view_live.py
Keys in the viewer: Space = pause, mouse = rotate / zoom.
The terminal prints the event ticker and KPIs.
"""
import sys, time
import mujoco, mujoco.viewer
from fleet_sim import FleetSim
from make_video import overlays

policy = sys.argv[1] if len(sys.argv) > 1 else "learned"
peer = not (len(sys.argv) > 2 and sys.argv[2] == "off")
sim = FleetSim(policy, peer_check=peer, seed=int(time.time()) % 1000)
last = None
with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
    v.cam.lookat[:] = [0.5, 0.5, 0]; v.cam.distance = 27; v.cam.elevation = -58; v.cam.azimuth = 90
    while v.is_running():
        t0 = time.time()
        sim.step(2)
        with v.lock():
            v.user_scn.ngeom = 0
            overlays(sim, v.user_scn)
        v.sync()
        if sim.ticker and sim.ticker[0] is not last:
            last = sim.ticker[0]
            m = sim.metrics()
            print(f"[{last[0]:6.1f}s] {last[1]:<55}  resolved {m['resolved']}/{m['requests']}  "
                  f"msgs {m['sends']}  blocked {m['blocked']}  unsafe {m['unsafe']}")
        time.sleep(max(0, 2 * sim.model.opt.timestep - (time.time() - t0)))
