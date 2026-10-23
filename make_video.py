"""Side-by-side MP4 renders (offscreen, OSMesa/EGL).

  python make_video.py routing   # broadcast vs learned routing, same seed
  python make_video.py peer      # learned routing, peer check off vs on

Headless Linux:  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python make_video.py routing
On a Mac the default (glfw/cgl) works.
"""
import os, sys, subprocess
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fleet_sim import FleetSim, COLORS

W, H, FPS = 640, 400, 20
SECONDS = float(os.environ.get("SECONDS", 40))
PKT = {"req": (1, 1, 1, 1), "fwd": (0.4, 0.7, 1, 1), "chal": (1, 0.25, 0.2, 1)}


def font(sz, bold=False):
    for f in ([f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
               "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"]):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


F_T, F_S, F_K = font(17, True), font(12), font(13, True)


def add_geom(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        (np.eye(3) if mat is None else mat).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def add_line(scn, a, b, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def overlays(sim, scn):
    """Requests, messages in flight, rogue / challenge markers."""
    for j in sim.jobs:
        if j.stage in ("open", "going"):
            r, g, b = COLORS[j.cat]
            add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.7, 0.02, 0], [j.loc[0], j.loc[1], 0.03], [r, g, b, 0.55])
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.18, 0, 0], [j.loc[0], j.loc[1], 2.2], [r, g, b, 0.9])
    for p in sim.packets:
        if p["frm"] is None:
            continue
        a, b = sim.pos(p["frm"]) + [0, 0, 1.4], sim.pos(p["to"]) + [0, 0, 1.4]
        add_line(scn, a, b, 0.035, (*PKT[p["kind"]][:3], 0.35))
        f = min(1.0, max(0.0, (sim.t - p["t0"]) / (p["t1"] - p["t0"])))
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.16, 0, 0], a + f * (b - a), PKT[p["kind"]])
    for ag in sim.agents:
        q = sim.pos(ag)
        if ag.rogue > 0:
            add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [1.0, 0.03, 0], [q[0], q[1], 0.05], [1, 0.2, 0.15, 0.8])
        if ag.challenged > 0:
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.35, 0, 0], [q[0], q[1], q[2] + 2.0], [1, 0.25, 0.2, 0.95])
        if ag.task is not None and ag.goal is not None:
            add_line(scn, q + [0, 0, 0.1], [ag.goal[0], ag.goal[1], 0.1], 0.02, (*COLORS[ag.skill], 0.5))


def hud(img, sim, title, subtitle):
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 46], fill=(8, 12, 15, 210))
    d.text((10, 5), title, font=F_T, fill=(240, 244, 246))
    d.text((10, 27), subtitle, font=F_S, fill=(160, 172, 180))
    m = sim.metrics()
    kp = [("resolved", f"{m['resolved']}/{m['requests']}"),
          ("msgs", f"{m['sends']}"),
          ("wasted m", f"{m['wasted_travel_m']:.0f}"),
          ("blocked", f"{m['blocked']}"), ("unsafe", f"{m['unsafe']}")]
    x = 10
    d.rectangle([0, H - 30, W, H], fill=(8, 12, 15, 210))
    for k, v in kp:
        col = (255, 107, 74) if (k == "unsafe" and m["unsafe"]) else (46, 196, 182)
        d.text((x, H - 24), k, font=F_S, fill=(160, 172, 180)); x += d.textlength(k, font=F_S) + 5
        d.text((x, H - 25), v, font=F_K, fill=col); x += d.textlength(v, font=F_K) + 16
    d.text((W - 10, H - 24), f"t={sim.t:5.1f}s  SYNTHETIC", font=F_S, fill=(160, 172, 180), anchor="ra")
    y = 52
    for t, msg in sim.ticker[:4]:
        a = int(255 * max(0.35, 1 - (sim.t - t) / 8))
        d.text((10, y), msg, font=F_S, fill=(230, 236, 240, a)); y += 15


def render(pairs, out_path):
    sims = [FleetSim(**kw) for kw, _, _ in pairs]
    if os.environ.get("MUJOCO_GL") == "osmesa":          # software rendering: shadows cost 2x
        for s in sims:
            s.model.light_castshadow[:] = 0
    rends = [mujoco.Renderer(s.model, H, W) for s in sims]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.5, 0.5, 0]; cam.distance = 27; cam.elevation = -58; cam.azimuth = 90
    steps = int(round(1 / (FPS * sims[0].model.opt.timestep)))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W * len(sims) + 4 * (len(sims) - 1)}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", out_path], stdin=subprocess.PIPE)
    for fr in range(int(SECONDS * FPS)):
        tiles = []
        for s, r, (_, title, sub) in zip(sims, rends, pairs):
            s.step(steps)
            r.update_scene(s.data, camera=cam)
            overlays(s, r.scene)
            img = Image.fromarray(r.render())
            hud(img, s, title, sub)
            tiles.append(np.asarray(img.convert("RGB")))
        sep = np.full((H, 4, 3), 30, np.uint8)
        row = tiles[0]
        for t in tiles[1:]:
            row = np.concatenate([row, sep, t], axis=1)
        ff.stdin.write(row.tobytes())
    ff.stdin.close(); ff.wait()
    for s, (_, title, _) in zip(sims, pairs):
        print(title, {k: v for k, v in s.metrics().items() if k in ("resolved", "requests", "sends", "wasted_travel_m", "blocked", "unsafe")})


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "routing"
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    seed = int(os.environ.get("SEED", 3))
    if which == "routing":
        render([(dict(policy="broadcast", seed=seed), "Broadcast to everyone", "every able machine races; losers turn back"),
                (dict(policy="learned", seed=seed), "YOROSIX learned routing", "pass to the machine the help model ranks first")],
               os.path.join(HERE, "out", "fleet_routing.mp4"))
    else:
        render([(dict(policy="learned", peer_check=False, seed=seed, unsafe_every=7.0), "Peer check OFF", "unsafe command goes through"),
                (dict(policy="learned", peer_check=True, seed=seed, unsafe_every=7.0), "Peer check ON (CHALLENGE)", "a trusted teammate stops it")],
               os.path.join(HERE, "out", "fleet_peercheck.mp4"))
