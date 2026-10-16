"""Make the README's figures, into results/.

Run:  MUJOCO_GL=egl python results/make_figures.py [tracking] [robot] [cross_validation] [convergence] [invariants] [three_cm] [height] [model_io]
(no names: all eight)

  tracking          results/tracking.gif: one real push from clip 5, with the tracking drawn on
                    (needs data/tracked_clips/clip5_overlay.mp4, from track.py --overlay)
  robot             results/robot.gif: the simulated robot pushing the box onto a target (green
                    outline), choosing each push with models/video.pt
  cross_validation  results/cross_validation.png: from data/cross_validation.json
  convergence       results/convergence.png: distance to the target after each push, model vs rule
                    (needs results/plan_video_precision.json and plan_rule_precision.json, from
                    planner.py --within 0.1 0.1)
  invariants        results/invariants.png: the model's error against how many pushes it trained on, with
                    and without the invariants (needs results/invariants/*.json, from cross_validate.py)
  three_cm          results/three_cm.png: every push exactly 3 cm, distance and angle to the target after each
                    push on one task: the model looking one or two pushes ahead, the rule and the adapted rule
                    (needs results/plan_{video,rule,rule_offset}_3cm_strict.json and plan_video_3cm_strict_ahead2.json,
                    from planner.py --length 0.03 --within 5 3, with --ahead 2 for the last)
  height            results/height.png: a side view of why each mark is placed at its own height
  model_io          results/model_io.png: what the model sees and predicts in one 0.1 s step
                    (needs models/video.pt)
"""
import json
import os
import sys

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repo's top folder
sys.path[:0] = [os.path.join(ROOT, d) for d in ("video", "plan", "wm")]

# Chart colours: blue = the world model, orange = the hand-written rule, grey = how far the box really moved.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL, RULE, STILL = "#2a78d6", "#eb6834", "#83827c"


def save_gif(path, frames, keep=(), colors=64, duration=100):
    """A small GIF: one colour palette shared by all frames (rendered shading compresses badly otherwise),
    with slots kept for rare colours that must survive."""
    from PIL import Image
    sample = Image.fromarray(np.concatenate(frames[::max(1, len(frames) // 8)], 0))
    palette = sample.quantize(colors - len(keep), method=Image.Quantize.MEDIANCUT).getpalette()[:3 * (colors - len(keep))]
    palette += [c for rgb in keep for c in rgb]
    p = Image.new("P", (1, 1))
    p.putpalette(palette + [0] * (768 - len(palette)))
    images = [Image.fromarray(f).quantize(palette=p, dither=Image.Dither.NONE) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=True)


def need(*paths):
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: see the top of results/make_figures.py for what makes it")


def tracking(clip="clip5"):
    """A few seconds of the tracking overlay around one push that both slides and turns the box."""
    import pandas as pd
    from geometry import load_camera, project
    need("data/human_pushes.csv", f"data/tracked_clips/{clip}_overlay.mp4")
    df = pd.read_csv("data/human_pushes.csv")
    setup = json.load(open("data/setup.json"))
    pose = json.load(open(f"data/tracked_clips/{clip}.json"))
    cam = (*load_camera("data/camera.json"), np.array(pose["R"]), np.array(pose["t"]))

    for _, g in df[df["clip"] == clip].groupby("episode"):
        b = g[["box_x", "box_y", "box_yaw"]].to_numpy()
        moved, turned = 1000 * np.linalg.norm(b[-1, :2] - b[0, :2]), np.degrees(abs(b[-1, 2] - b[0, 2]))
        if 25 < moved < 45 and 8 < turned < 25:
            break
    else:
        raise SystemExit(f"no push in {clip} moves the box 25-45 mm and turns it 8-25 deg")
    start, length = g["start"].iloc[0], g["t"].iloc[-1]

    # Crop round the box's average position (the overlay video is half size).
    centre = project([[g["box_x"].mean(), g["box_y"].mean(), setup["box_height"]]], *cam)[0] / 2
    cap = cv2.VideoCapture(f"data/tracked_clips/{clip}_overlay.mp4")
    fps, W, H = cap.get(cv2.CAP_PROP_FPS), 560, 420
    x0 = int(np.clip(centre[0] - W / 2, 0, cap.get(3) - W))
    y0 = int(np.clip(centre[1] - H / 2, 0, cap.get(4) - H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((start - 0.3) * fps))
    frames = []
    for i in range(int((length + 1.0) * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if i % 3 == 0:  # about 10 frames a second
            frames.append(cv2.cvtColor(f[y0:y0 + H, x0:x0 + W], cv2.COLOR_BGR2RGB))
    imageio.mimsave("results/tracking.gif", frames, duration=100, loop=0)
    print(f"results/tracking.gif: {clip} at {start:.1f} s, the box moved {moved:.0f} mm and turned {turned:.0f} deg")


def robot():
    """The simulated robot pushing the box onto a target with the video-trained model."""
    from planner import load_model, random_task, run_task
    from push_env import BOX_SIZE, TABLE_Z, make_env, render
    need("models/video.pt")
    model = load_model("models/video.pt")[0]
    env = make_env(render=True, camera="pushview", size=(480, 360))

    tasks = np.random.default_rng(1000)
    while True:  # a target with a real turn and some distance, so the GIF shows both
        start, target = random_task(tasks)
        turn = abs((target[2] - start[2] + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        if np.linalg.norm(target[:2] - start[:2]) > 0.05 and np.degrees(turn) > 25:
            break

    # The target's top outline, projected into the camera's picture.
    cid = env.sim.model.camera_name2id(env.render_camera)
    pos, R = env.sim.data.cam_xpos[cid], env.sim.data.cam_xmat[cid].reshape(3, 3)
    w, h = env.render_size
    f = (h / 2) / np.tan(np.radians(env.sim.model.cam_fovy[cid]) / 2)
    c, s, a = np.cos(target[2]), np.sin(target[2]), BOX_SIZE[0] / 2
    outline = []
    for dx, dy in ((a, a), (-a, a), (-a, -a), (a, -a)):
        q = R.T @ (np.array([target[0] + c * dx - s * dy, target[1] + s * dx + c * dy, TABLE_Z + BOX_SIZE[2]]) - pos)
        outline.append([w / 2 + f * q[0] / -q[2], h / 2 - f * q[1] / -q[2]])  # camera looks along its -z
    outline = np.array(outline, np.int32)

    frames, count = [], {"travel": 0, "push": 0}

    def grab(e, phase):
        """Every 2nd step while pushing (real time), every 6th while travelling (3x speed)."""
        count[phase] += 1
        if count[phase] % (2 if phase == "push" else 6) == 0:
            frame = np.ascontiguousarray(render(e))
            cv2.polylines(frame, [outline], True, (40, 200, 90), 2)
            frames.append(cv2.resize(frame, (360, 270), interpolation=cv2.INTER_AREA))

    r = run_task(env, "model", start, target, np.random.default_rng(0), model, on_step=grab)
    env.close()  # else its renderer is cleaned up after the graphics context, with a scary message
    # keep the target outline's green and the ball's orange (lit to dark)
    save_gif("results/robot.gif", frames + [frames[-1]] * 15,
             keep=[(40, 200, 90), (255, 128, 26), (200, 100, 20), (140, 70, 15)])
    print(f"results/robot.gif: target {100 * np.linalg.norm(target[:2] - start[:2]):.1f} cm away and turned "
          f"{np.degrees(turn):.0f} deg, {'reached' if r['success'] else 'missed'} after {r['pushes']} pushes")


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def finish(fig, axes, title, path):
    top = 0.93 if title else 1.0  # room for the title, if there is one
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper left",
               bbox_to_anchor=(0.005, top), ncols=2, fontsize=9)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86 if title else 0.89))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(path)


def cross_validation():
    """The model's error per left-out clip, next to how far the box really moved."""
    need("data/cross_validation.json")
    d = json.load(open("data/cross_validation.json"))
    clips = [k for k in d if k != "average"]
    rows = [d[c] for c in clips] + [d["average"]]
    x, w = np.arange(len(rows)), 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for ax, (model, still, unit, title) in zip(axes, [("pos_err_mm", "still_pos_mm", "mm", "Position"),
                                                      ("yaw_err_deg", "still_yaw_deg", "°", "Angle")]):
        m, s = [r[model] for r in rows], [r[still] for r in rows]
        ax.bar(x - w / 2 - 0.01, m, w, color=MODEL, label="model's error", zorder=3)
        ax.bar(x + w / 2 + 0.01, s, w, color=STILL, label="how far the box really moved", zorder=3)
        for dx, values in ((-w / 2 - 0.01, m), (w / 2 + 0.01, s)):
            for xi, v in zip(x, values):
                ax.text(xi + dx, v + max(s) * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x, [c.replace("clip", "clip ") for c in clips] + ["average"])
        ax.axvline(len(clips) - 0.5, color=GRID, linewidth=1)
        style(ax, title, f"median per push ({unit})")
    finish(fig, axes, "Predicting pushes from a clip the model never saw (trained on the other 4 clips)",
           "results/cross_validation.png")


def convergence():
    """Distance to the target after each push, model vs rule: median over the targets, band = middle half."""
    paths = {"video": "results/plan_video_precision.json", "rule": "results/plan_rule_precision.json"}
    need(*paths.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    ends = ([], [])  # each panel's last median per policy, labelled at the end of its line
    for name, colour, label in (("video", MODEL, "world model (learned from my finger)"), ("rule", RULE, "hand-written rule")):
        pos, ang = [], []
        for r in json.load(open(paths[name]))["results"]:
            p, t = np.array(r["poses"]), np.array(r["target"])
            pos.append(1000 * np.linalg.norm(p[:, :2] - t[:2], axis=1))
            ang.append(np.degrees(np.abs((p[:, 2] - t[2] + np.pi / 4) % (np.pi / 2) - np.pi / 4)))
        n = max(len(e) for e in pos)  # a run that stopped early keeps its last error
        for ax, e, end in zip(axes, (pos, ang), ends):
            e = np.array([np.pad(v, (0, n - len(v)), mode="edge") for v in e])
            lo, med, hi = np.percentile(e, [25, 50, 75], axis=0)
            ax.fill_between(np.arange(n), lo, hi, color=colour, alpha=0.15, linewidth=0)
            ax.plot(np.arange(n), med, color=colour, linewidth=2, marker="o", markersize=4, label=label)
            end.append(med[-1])
    for ax, (a, b) in zip(axes, ends):  # the higher label a little up, the lower a little down
        for v, other in ((a, b), (b, a)):
            ax.annotate(f"{v:.1f}", (n - 1, v), xytext=(6, 5 if v >= other else -5), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
    for ax, unit, title, ticks in zip(axes, ("mm", "°"), ("Distance to the target", "Angle to the target"),
                                      ([2, 5, 10, 20, 50], [0.5, 1, 2, 5, 10, 20])):
        ax.set_yscale("log")
        ax.set_yticks(ticks)
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("pushes so far")
        style(ax, title, f"median ({unit})")
    finish(fig, axes, "",
           "results/convergence.png")


def invariants():
    """Leave-one-clip-out error against the number of training pushes, for each version of the model."""
    versions = (("all", MODEL, "o", "both"),
                ("no-copies", "#1baf7a", "s", "box frame only"),
                ("no-box-frame-copies", "#c2457a", "D", "copies only"),
                ("no-box-frame", "#eda100", "^", "neither"))
    sizes = ("10", "20", "40", "80", "all")
    paths = [f"results/invariants/{v}_{n}.json" for v, *_ in versions for n in sizes]
    need(*paths)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for v, colour, marker, label in versions:
        runs = [json.load(open(f"results/invariants/{v}_{n}.json")) for n in sizes]
        folds = [k for k in runs[0] if k.startswith("clip")]
        total = sum(runs[0][c]["pushes"] for c in folds)
        # "all" trains on every push of the other clips: how many that is, on average over the folds
        x = [int(n) for n in sizes[:-1]] + [np.mean([total - runs[0][c]["pushes"] for c in folds])]
        for ax, key in zip(axes, ("pos_err_mm", "yaw_err_deg")):
            y = [r["average"][key] for r in runs]
            ax.plot(x, y, color=colour, linewidth=2, marker=marker, markersize=6, label=label, zorder=3)
            ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
    still = runs[-1]["average"]
    for ax, key, unit, title, ticks in zip(axes, ("still_pos_mm", "still_yaw_deg"), ("mm", "°"), ("Position", "Angle"),
                                           ([2, 5, 10, 20, 50], [1, 2, 5, 10, 20])):
        ax.axhline(still[key], color=STILL, linewidth=1.5, linestyle="--", zorder=2)
        ax.annotate("how far the box really moved", (x[-1], still[key]), xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([10, 20, 40, 80, round(x[-1])])
        ax.set_yticks(ticks)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("pushes trained on")
        style(ax, title, f"median error per push ({unit})")
    finish(fig, axes, "",
           "results/invariants.png")


def three_cm(task=0):
    """Every push exactly 3 cm: distance and angle to the target after each push on one task (the first).
    Top: the model looking one push ahead vs the rule. Bottom: each improved, the model looking two pushes
    ahead vs the rule adapted by hand to fixed-length pushes."""
    rows = ((("video_3cm_strict", MODEL, "world model"), ("rule_3cm_strict", RULE, "hand-written rule")),
            (("video_3cm_strict_ahead2", MODEL, "world model, two pushes ahead"),
             ("rule_offset_3cm_strict", RULE, "rule adapted to fixed-length pushes")))
    paths = {name: f"results/plan_{name}.json" for row in rows for name, *_ in row}
    need(*paths.values())
    pos_ok, angle_ok = json.load(open(paths["video_3cm_strict"]))["within_mm_deg"]
    runs = {name: json.load(open(path))["results"] for name, path in paths.items()}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE)
    for row, (ax_d, ax_a), heading in zip(rows, axes, ("Looking one push ahead", "Improved: the model looks two pushes ahead, the rule is adapted")):
        for name, colour, label in row:
            r = runs[name][task]
            pose = np.array(r["poses"])
            dist = 1000 * np.linalg.norm(pose[:, :2] - np.array(r["target"][:2]), axis=1)
            angle = np.degrees((pose[:, 2] - r["target"][2] + np.pi / 4) % (np.pi / 2) - np.pi / 4)
            for ax, y in ((ax_d, dist), (ax_a, angle)):
                ax.plot(np.arange(len(y)), y, color=colour, linewidth=2, marker="o", markersize=4, label=label, zorder=3)
        for ax, lo, hi, title, ylabel in ((ax_d, 0, pos_ok, f"{heading}\nDistance to the target (grey: success)", "mm"),
                                          (ax_a, -angle_ok, angle_ok, "\nAngle off the target (grey: success)", "degrees")):
            ax.axhspan(lo, hi, color=GRID, linewidth=0, zorder=1)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xlim(-0.5, 15.5)
            ax.set_xlabel("pushes so far")
            style(ax, title, ylabel)
        ax_a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        ax_d.set_ylim(0, 75)    # room for the legend above the lines
        ax_d.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout(h_pad=2)
    fig.savefig("results/three_cm.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("results/three_cm.png")


def height():
    """Side view, real proportions: a pixel only gives a line from the camera; the dot's known height picks the point."""
    import json as _  # noqa: F401  (nothing to load: the numbers are the setup's)
    cam_h, box_h, box_x, box_w, dot_x = 47.0, 5.7, 14.0, 12.8, 20.0           # cm
    floor_x = dot_x * cam_h / (cam_h - box_h)                                 # where the line meets the floor
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.axhline(0, color=INK2, linewidth=1.5)
    ax.add_patch(matplotlib.patches.Rectangle((box_x, 0), box_w, box_h, facecolor="#cde2fb", edgecolor=MODEL, linewidth=1.5))
    ax.plot([0, dot_x], [cam_h, box_h], color=INK, linewidth=1.5)
    ax.plot([dot_x, floor_x], [box_h, 0], color=INK, linewidth=1.5, linestyle=":")
    ax.plot(0, cam_h, marker="v", markersize=14, color=INK)
    ax.plot(dot_x, box_h, "o", markersize=9, color="#1baf7a", markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.plot(floor_x, 0, "x", markersize=9, markeredgewidth=2, color=STILL, zorder=4)
    ax.annotate("camera", (0, cam_h), xytext=(14, 6), textcoords="offset points", va="bottom", fontsize=9, color=INK)
    ax.annotate("one pixel", (dot_x * 0.45, cam_h - (cam_h - box_h) * 0.45), xytext=(10, 0), textcoords="offset points",
                fontsize=9, color=INK)
    ax.annotate(f"dot, {box_h:g} cm up", (dot_x, box_h), xytext=(40, 45), textcoords="offset points", fontsize=9,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.annotate(f"if on the floor: {floor_x - dot_x:.1f} cm off", (floor_x, 0), xytext=(4, -16),
                textcoords="offset points", ha="right", va="center", fontsize=9, color=INK2)
    ax.annotate("box", (box_x + 2.5, box_h / 2), ha="center", va="center", fontsize=9, color=MODEL)
    ax.annotate("floor", (-1.5, 0), xytext=(0, 4), textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlim(-2, 36)
    ax.set_ylim(-6, 55)
    ax.set_xlabel("cm along the floor")
    ax.set_ylabel("cm above the floor")
    style(ax, "", "cm above the floor")
    fig.suptitle("Why each mark's height matters", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("results/height.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("results/height.png")


def model_io(clip="clip5", mag=5):
    """One 0.1 s step, in the box's own frame: what the model sees (where the finger is, and its move) and what it
    predicts (the box's slide X, Y and turn θ), on the push of tracking.gif. Motion is drawn mag times bigger
    (the README's caption says so)."""
    import pandas as pd
    import torch
    from model import load_model, steps, turn
    need("data/human_pushes.csv", "models/video.pt")
    df = pd.read_csv("data/human_pushes.csv")
    for _, g in df[df["clip"] == clip].groupby("episode"):  # the push of tracking.gif
        b = g[["box_x", "box_y", "box_yaw"]].to_numpy()
        moved, turned = 1000 * np.linalg.norm(b[-1, :2] - b[0, :2]), np.degrees(abs(b[-1, 2] - b[0, 2]))
        if 25 < moved < 45 and 8 < turned < 25:
            break
    X, Y, _ = steps(g)
    k = np.argmax(np.linalg.norm(Y[:, :2], axis=1))  # the step where the box really moved most
    with torch.no_grad():
        P = load_model("models/video.pt")[0](torch.tensor(X[k:k + 1], dtype=torch.float32)).mean(0)[0].numpy()
    h = 500 * json.load(open("data/setup.json"))["box_size"]  # half the box, mm
    finger, move = 1000 * X[k, :2], 1000 * mag * X[k, 2:]
    slide, spin = 1000 * mag * P[:2], np.degrees(mag * P[2])
    FINGER = "#1baf7a"
    fig, ax = plt.subplots(figsize=(4.4, 4), facecolor=SURFACE)

    def square(centre, deg, **kw):
        corners = turn(np.array([[h, h], [-h, h], [-h, -h], [h, -h]]), np.radians(deg)) + centre
        ax.add_patch(matplotlib.patches.Polygon(corners, closed=True, linewidth=1.5, zorder=2, **kw))
        return corners

    def arrow(start, end, colour, **kw):
        ax.annotate("", end, start, arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.5, shrinkA=0,
                                                    shrinkB=0, **kw), zorder=4)

    BOX = "#eeedea"
    label = dict(textcoords="offset points", bbox=dict(boxstyle="square,pad=0.15", facecolor=BOX, edgecolor="none"))
    square((0, 0), 0, facecolor=BOX, edgecolor=INK2)
    o = np.array([-h, -1.45 * h])  # the box's own axes, drawn beside it
    for d, name in (((0.25 * h, 0), "x"), ((0, 0.25 * h), "y")):
        arrow(o, o + d, INK2)
        ax.annotate(name, o + d, xytext=(3, 3), textcoords="offset points", color=INK2)

    # Inputs: where the finger is, and its move in the next 0.1 s.
    arrow((0, 0), finger, INK, linestyle=(0, (3, 2)))
    ax.annotate("(x, y)", finger / 2, xytext=(0, 6), ha="center", va="bottom", **label)
    arrow(finger, finger + move, FINGER)
    ax.annotate("(x', y')", finger + move, xytext=(6, 0), va="center", color=FINGER, **label)
    ax.plot(*finger, "o", markersize=9, color=FINGER, markeredgecolor=SURFACE, zorder=5)

    # Output: where the box goes in that 0.1 s.
    corners = square(slide, spin, fill=False, edgecolor=MODEL, linestyle="--")
    arrow((0, 0), slide, MODEL)
    ax.annotate("(X, Y)", slide / 2, xytext=(0, -6), ha="center", va="top", color=MODEL, **label)
    r = 0.8 * h  # θ: the box's x direction before (dotted) and after (blue)
    ax.plot(*np.array([slide, slide + (r, 0)]).T, color=INK2, linewidth=1, linestyle=":", zorder=3)
    ax.plot(*np.array([slide, slide + turn(np.array([r, 0.0]), np.radians(spin))]).T, color=MODEL, linewidth=1.2, zorder=3)
    ax.add_patch(matplotlib.patches.Arc(slide, 1.3 * r, 1.3 * r, theta1=min(0, spin), theta2=max(0, spin),
                                        color=MODEL, linewidth=1.5, zorder=3))
    ax.annotate("θ", slide + turn(np.array([0.65 * r, 0.0]), np.radians(spin / 2)), xytext=(14, 0),
                textcoords="offset points", va="center", color=MODEL)
    top = corners[np.argmax(corners[:, 1])]
    ax.annotate("model's prediction (X, Y, θ)", top, xytext=(0, 6), textcoords="offset points", ha="center",
                va="bottom", color=MODEL)

    ax.update_datalim([finger, finger + move])
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.06)
    ax.axis("off")
    fig.savefig("results/model_io.png", dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("results/model_io.png")

if __name__ == "__main__":
    os.chdir(ROOT)
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    figures = dict(tracking=tracking, robot=robot, cross_validation=cross_validation, convergence=convergence,
                   invariants=invariants, three_cm=three_cm, height=height, model_io=model_io)
    names = sys.argv[1:] or list(figures)
    unknown = [n for n in names if n not in figures]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}: choose from {list(figures)}")
    os.makedirs("results", exist_ok=True)
    for name in names:
        figures[name]()
