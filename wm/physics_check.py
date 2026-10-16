"""
Experiment 3: Did the model rediscover the physics? The same simple pushes, done by the robot in the simulator
and predicted by the world model trained on my finger, side by side.

MUJOCO_GL=egl python wm/physics_check.py models/video.pt --out results/physics_check.json \
          --plot results/physics_check.png
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plan"))
from planner import SPEED, tip_path  # noqa: E402  (planner also puts sim/ on the path)
from model import load_model, rollout  # noqa: E402
from push_env import BOX_SIZE, make_env, push  # noqa: E402

SIDE, LENGTH, START = 2, 0.03, np.array([0.02, 0.0, 0.0])  # side 2 faces the robot; the box starts square on
PUSHES = [  # (where on the side: 0 = middle, +-1 = a corner; angle in degrees; name)
    (0.0, 0, "middle"), (0.5, 0, "halfway to a corner"), (-0.5, 0, "halfway, other way"),
    (1.0, 0, "corner"), (-1.0, 0, "other corner"), (0.0, 30, "middle, 30°"), (0.0, -30, "middle, -30°")]


def compare(model):
    """The PUSHES, done in the simulator and predicted by the model."""
    env = make_env()
    steps = int(np.ceil((0.02 + LENGTH) / (SPEED * 0.1))) + 3   # the whole push, then a stop
    rows = []
    for offset, angle, name in PUSHES:
        env.set_box_pose(*START)
        for _ in range(5):
            env.step(np.array([0, 0, 0, 1.0]))
        before = env.box_pose()
        push(env, SIDE, offset, np.radians(angle), LENGTH, speed=SPEED)
        sim = env.box_pose() - before
        tips = tip_path(START, SIDE, offset, np.radians(angle), LENGTH, steps)
        pred = rollout(model, START[None], tips[None]).numpy()[:, 0, -1].mean(0) - START
        for d in (sim, pred):
            d[2] = (d[2] + np.pi) % (2 * np.pi) - np.pi
        rows.append(dict(push=name, offset=offset, angle_deg=angle, sim=sim.tolist(), model=pred.tolist()))
    env.close()
    return rows


def gaps(rows):
    """How far the model and the simulator disagree, on average over the pushes: position (mm), turn (degrees)."""
    pos = np.mean([1000 * np.linalg.norm(np.subtract(r["model"][:2], r["sim"][:2])) for r in rows])
    turn = np.mean([np.degrees(abs(r["model"][2] - r["sim"][2])) for r in rows])
    return float(pos), float(turn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--plot")
    args = parser.parse_args()
    model = load_model(args.model)[0]

    rows = compare(model)
    print(f"{'push':22s} {'simulator: forward, sideways, turn':>36s}   {'model: forward, sideways, turn':>32s}")
    for r in rows:
        s, m = np.array(r["sim"]), np.array(r["model"])
        print(f"{r['push']:22s} {1000 * s[0]:10.1f} mm {1000 * s[1]:6.1f} mm {np.degrees(s[2]):6.1f}°"
              f"   {1000 * m[0]:10.1f} mm {1000 * m[1]:6.1f} mm {np.degrees(m[2]):6.1f}°")
    pos, turn = gaps(rows)
    print(f"on average the model and the simulator are {pos:.1f} mm and {turn:.1f}° apart")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    if args.plot:
        plot(rows, args.plot)


def plot(rows, path):
    """Seen from above: the box before (grey), after the simulator's push (black) and as the model predicted (blue)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = BOX_SIZE[0] / 2
    square = np.array([[h, h], [-h, h], [-h, -h], [h, -h], [h, h]])

    def outline(pose):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return 100 * (square @ np.array([[c, s], [-s, c]]) + pose[:2])

    cols = (len(rows) + 2) // 2  # two lines; a free spot at the end holds the legend
    fig, axes = plt.subplots(2, cols, figsize=(2.3 * cols, 6.2), facecolor="#fcfcfb")
    axes = axes.ravel()
    for ax in axes[len(rows):]:
        ax.axis("off")
    for ax, r in zip(axes, rows):
        ax.plot(*outline(np.zeros(3)).T, color="#c3c2b7", linewidth=1.5)
        ax.plot(*outline(np.array(r["sim"])).T, color="#0b0b0b", linewidth=2, label="simulator")
        ax.plot(*outline(np.array(r["model"])).T, color="#2a78d6", linewidth=2, linestyle="--", label="model")
        s, m = np.array(r["sim"]), np.array(r["model"])
        ax.set_title(f"{r['push']}\nsim {1000 * s[0]:.1f}, {1000 * s[1]:+.1f} mm, {np.degrees(s[2]):+.1f}°\n"
                     f"model {1000 * m[0]:.1f}, {1000 * m[1]:+.1f} mm, {np.degrees(m[2]):+.1f}°", fontsize=8.5, color="#0b0b0b")
        ax.set_aspect("equal")
        ax.set_xlim(-9.5, 12.5)
        ax.set_ylim(-11, 11)
        ax.axis("off")
    axes[-1].legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="center", bbox_to_anchor=(0.5, 0.6))
    axes[-1].text(0.5, 0.3, "numbers: forward, sideways, turn", ha="center", transform=axes[-1].transAxes,
                  fontsize=8.5, color="#52514e")
    fig.suptitle("The same 3 cm pushes (from the left): the robot in the simulator vs the model learned from my finger",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
