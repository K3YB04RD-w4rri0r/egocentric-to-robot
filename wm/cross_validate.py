""" Cross validation: (Training, Experiment 1 and Experiment 2)

python wm/cross_validate.py data/human_pushes.csv --out data/cross_validation.json


To see what each invariant is worth (results/invariants/):
  --without copies     each push shown once, not also turned and mirrored (8 versions)
  --without box-frame  the model sees raw floor positions, as the camera does (and gets no copies)
  --with-copies        with --without box-frame: still give it the 8 copies
  --pushes N           train on only N randomly chosen pushes of the other clips
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from model import WorldModel, fit, rollout, steps, wrap

SHOWN = ("pos_err_mm", "yaw_err_deg", "still_pos_mm", "still_yaw_deg")


def push_errors(model, df):
    """Per test push: the model's final position and angle error, how far the box really moved, and
    the networks' disagreement about the final pose."""
    rows = []
    groups = [g for _, g in df.groupby("episode", sort=False)]
    for T in sorted({len(g) for g in groups}):           # pushes of the same length go together
        same = [g for g in groups if len(g) == T]
        box = np.stack([g[["box_x", "box_y", "box_yaw"]].to_numpy() for g in same])
        tip = np.stack([g[["tip_x", "tip_y"]].to_numpy() for g in same])
        end = rollout(model, box[:, 0], tip).numpy()[:, :, -1]  # members x B x 3
        mean = end.mean(0)
        for b in range(len(same)):
            rows.append(dict(
                pos_err=1000 * np.linalg.norm(mean[b, :2] - box[b, -1, :2]),
                yaw_err=np.degrees(abs(wrap(mean[b, 2] - box[b, -1, 2]))),
                moved=1000 * np.linalg.norm(box[b, -1, :2] - box[b, 0, :2]),
                turned=np.degrees(abs(wrap(box[b, -1, 2] - box[b, 0, 2]))),
                spread_pos=1000 * np.linalg.norm(end[:, b, :2].std(0))))
    return rows


def summary(rows):
    """Medians over the pushes, and the error on the half the networks agree on most against the other half."""
    r = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    confident = r["spread_pos"] <= np.median(r["spread_pos"])
    return dict(pushes=len(rows),
                pos_err_mm=float(np.median(r["pos_err"])), yaw_err_deg=float(np.median(r["yaw_err"])),
                still_pos_mm=float(np.median(r["moved"])), still_yaw_deg=float(np.median(r["turned"])),
                pos_err_confident_half_mm=float(np.median(r["pos_err"][confident])),
                pos_err_doubtful_half_mm=float(np.median(r["pos_err"][~confident])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--without", choices=["copies", "box-frame"])
    parser.add_argument("--with-copies", action="store_true", help="with --without box-frame: keep the 8 copies")
    parser.add_argument("--pushes", type=int, default=0, help="train on only this many pushes (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    rng = np.random.default_rng(args.seed)
    floor = args.without == "box-frame"
    results = {}
    for clip in sorted(df["clip"].unique()):
        train = df[df["clip"] != clip]
        if args.pushes:
            pushes = train["episode"].unique()
            train = train[train["episode"].isin(rng.choice(pushes, min(args.pushes, len(pushes)), replace=False))]
        X, Y, push = steps(train, floor)
        model = fit(WorldModel(floor=floor), X, Y, push, train_steps=args.train_steps, seed=args.seed,
                    copies=args.without is None or args.with_copies, log=lambda _: None)
        s = results[clip] = summary(push_errors(model, df[df["clip"] == clip]))
        print(f"{clip} left out ({s['pushes']} pushes): median error {s['pos_err_mm']:.1f} mm, {s['yaw_err_deg']:.1f} deg"
              f"   [box really moved: {s['still_pos_mm']:.1f} mm, {s['still_yaw_deg']:.1f} deg]", flush=True)

    folds = list(results.values())
    results["average"] = {k: float(np.mean([f[k] for f in folds])) for k in SHOWN}
    results["settings"] = dict(without=args.without, with_copies=args.with_copies, pushes=args.pushes, seed=args.seed)
    a = results["average"]
    print(f"average over the {len(folds)} clips: {a['pos_err_mm']:.1f} mm, {a['yaw_err_deg']:.1f} deg"
          f"   [box really moved: {a['still_pos_mm']:.1f} mm, {a['still_yaw_deg']:.1f} deg]")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
