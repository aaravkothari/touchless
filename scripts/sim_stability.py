"""Offline stability harness for the hand pointer pipeline.

Feeds HandPointerPipeline a synthetic fingertip track — a perfectly still
finger corrupted by landmark noise (Gaussian wobble + occasional spike
frames + slow drift), plus one deliberate move — and reports what the OS
cursor would have done. Lets stability tuning happen without a camera and
a steady hand:

    python scripts/sim_stability.py

Noise scales (per-frame std of the pointer signal):
  hand mode   pointer is in normalized camera coords; a still fingertip on
              a 640x480 webcam jitters ~0.001-0.004.
  wrist mode  pointer_rel is (tip - forearm ref) / hand size; the ref mixes
              two landmarks and the division renormalizes, so the same
              camera noise lands ~4-10x larger here.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from touchless.config import Config  # noqa: E402
from touchless.pipeline import HandPointerPipeline  # noqa: E402

SCREEN_W, SCREEN_H = 1920, 1080
FPS = 30.0
SPIKE_P = 0.04        # fraction of frames that are spike outliers
SPIKE_MAG = 5.0       # spike magnitude, in units of sigma
DRIFT_STD_PER_S = 0.004  # slow landmark drift ("the dots recalibrating")

# (phase name, duration s). still1 = the core complaint: finger dead still.
PHASES = [("still1", 10.0), ("move", 0.4), ("still2", 5.0), ("drift", 6.0)]


def make_track(rng, sigma, step):
    """(t, pointer, phase) arrays for one synthetic session."""
    ts, pts, phases = [], [], []
    t = 0.0
    base = np.zeros(2)
    drift = np.zeros(2)
    for name, dur in PHASES:
        n = int(dur * FPS)
        for i in range(n):
            dt = (1.0 / FPS) * rng.uniform(0.85, 1.15)  # frame-time jitter
            t += dt
            if name == "move":
                base = base + np.array(step) / n  # linear glide
            if name == "drift":
                drift = drift + rng.normal(0, DRIFT_STD_PER_S * dt ** 0.5, 2)
            p = base + drift + rng.normal(0, sigma, 2)
            if rng.random() < SPIKE_P:
                p = p + rng.normal(0, SPIKE_MAG * sigma, 2)
            ts.append(t)
            pts.append(p)
            phases.append(name)
    return np.array(ts), np.array(pts), np.array(phases)


def run(cfg, wrist, sigma, step, seed=7):
    rng = np.random.default_rng(seed)
    ts, pts, phases = make_track(rng, sigma, step)
    pipe = HandPointerPipeline(cfg, wrist, SCREEN_W, SCREEN_H)
    px = np.zeros((len(ts), 2))     # cursor position in screen px per frame
    moved = np.zeros(len(ts), bool)
    locked = np.zeros(len(ts), bool)
    cur = None
    for i, (t, p) in enumerate(zip(ts, pts)):
        sx, sy, m = pipe.update(p.copy(), float(t))
        if m or cur is None:
            cur = (sx * SCREEN_W, sy * SCREEN_H)
        px[i] = cur
        moved[i] = m
        locked[i] = pipe.still
    return ts, px, moved, locked, phases


def still_stats(ts, px, moved, locked, phases, name, warmup_s=1.5):
    sel = phases == name
    t0 = ts[sel][0] + warmup_s  # let filters converge before judging
    sel &= ts > t0
    p = px[sel]
    center = np.median(p, axis=0)
    dev = np.linalg.norm(p - center, axis=1)
    dur = ts[sel][-1] - t0
    return {
        "locked%": 100.0 * np.mean(locked[sel]),
        "moves/s": np.sum(moved[sel]) / dur,
        "rms_px": float(np.sqrt(np.mean(dev ** 2))),
        "p2p_px": float(np.max(np.abs(p - center), axis=0).max() * 2),
    }


def settle_time(ts, px, phases):
    """Seconds from move start until the cursor stays within 15 px of its
    final still2 position (responsiveness guard: smoothing must not lag)."""
    final = np.median(px[phases == "still2"][-60:], axis=0)
    t_start = ts[phases == "move"][0]
    tail = ts >= t_start
    dev = np.linalg.norm(px[tail] - final, axis=1)
    ok = dev < 15.0
    for i in range(len(ok)):
        if ok[i:].all() or (ok[i] and ts[tail][i] > t_start + 3.0):
            return float(ts[tail][i] - t_start)
    return float("inf")


def report(label, cfg, wrist, sigmas, step):
    print(f"\n=== {label} ===")
    hdr = (f"{'sigma':>7} | {'locked%':>7} {'moves/s':>8} {'rms_px':>7} "
           f"{'p2p_px':>7} | {'locked%':>7} {'drift_px':>8} | {'settle_s':>8}")
    print(hdr)
    print(f"{'':>7} | {'-- still (finger dead still) --':^32} "
          f"| {'-- drift --':^18} | {'move':>8}")
    for sigma in sigmas:
        ts, px, moved, locked, phases = run(cfg, wrist, sigma, step)
        s1 = still_stats(ts, px, moved, locked, phases, "still1")
        dr = still_stats(ts, px, moved, locked, phases, "drift")
        st = settle_time(ts, px, phases)
        print(f"{sigma:7.4f} | {s1['locked%']:7.1f} {s1['moves/s']:8.2f} "
              f"{s1['rms_px']:7.1f} {s1['p2p_px']:7.1f} "
              f"| {dr['locked%']:7.1f} {dr['p2p_px']:8.1f} | {st:8.2f}")


def main():
    cfg = Config()
    report("hand mode (pointer = index tip, cam coords)", cfg, wrist=False,
           sigmas=(0.001, 0.002, 0.004), step=(-0.08, 0.05))
    report("hand-wrist mode (pointer_rel, hand-size units)", cfg, wrist=True,
           sigmas=(0.005, 0.010, 0.020), step=(-0.20, 0.12))


if __name__ == "__main__":
    main()
