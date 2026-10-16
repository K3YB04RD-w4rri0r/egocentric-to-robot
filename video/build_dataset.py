"""Turn tracked clips into pushes, in the same format as the robot's pushes.

Run:  python video/build_dataset.py data/tracked_clips/*.csv --setup data/setup.json \
          --out data/human_pushes.csv

For each clip:
1. smooth the tracks and resample them to 10 samples per second, like the robot's log;
2. find pushes: moments when the box moves while your fingertip is right next to it.
   Keep up to 1 s before (the approach) and 0.3 s after (the stop), like the robot's log,
   but only while the finger is down (not while it is being lowered or lifted);
3. drop pushes where the fingertip or box was lost, the box was moved by hand (fingertip
   too far away), or the box kept gliding after your finger stopped.
Then, over all clips:
4. correct for the nail dot sitting behind the point that touches the box. When the box
   starts moving, the gap between the pusher and the box's side should be the same for
   your finger as for the robot's ball (ROBOT_GAP). The difference between your typical
   gap and the robot's is the correction, applied along each push's direction.

Output columns (metres, radians): episode, t, tip_x, tip_y, box_x, box_y, box_yaw, clip, start
(one episode = one push; start is when it begins in its clip, in seconds).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HALF = 0.064          # half the box's width (metres); set from setup.json in main
STEP = 0.1            # seconds between samples, as in the robot's log
MOVING = 0.0008       # the box counts as moving above 0.8 mm or ...
TURNING = np.radians(0.4)  # ... 0.4 degrees per 0.1 s (tracking noise is far below both)
STILL = 0.0015        # the fingertip counts as still below 1.5 mm per 0.1 s
ROBOT_GAP = 0.008     # robot's ball centre to box side when it touches: the ball's 8 mm radius
                      # (measured from 10 Hz snapshots: 8.6 mm; the difference is below tracking noise)


def wrap(a, period):
    """An angle difference brought into [-period/2, period/2)."""
    return (a + period / 2) % period - period / 2


def gap_to_box(tip, box):
    """Distance from the pusher to the box's outline (0 on the outline).

    tip: N x 2, box: N x 3 (x, y, yaw). Works for any yaw because a square repeats every 90°.
    """
    d = tip - box[:, :2]
    c, s = np.cos(box[:, 2]), np.sin(box[:, 2])
    local = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])
    return np.linalg.norm(np.maximum(np.abs(local) - HALF, 0), axis=1)


def runs(mask):
    """(start, end) index pairs of the stretches where mask is True (end exclusive)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def resample(track, columns, fps):
    """Smooth each unbroken stretch of the given columns and sample it every 0.1 s.

    Gaps of up to 3 frames are filled in; longer ones break the stretch.
    Returns a 10 Hz table with NaN where the columns weren't tracked.
    """
    t = track.t.to_numpy()
    grid = np.arange(t[0], t[-1], STEP)
    out = np.full((len(grid), len(columns)), np.nan)
    values = track[columns].interpolate(limit=3, limit_area="inside").to_numpy()
    window = max(5, int(0.2 * fps) | 1)  # about 0.2 s, odd
    for a, b in runs(~np.isnan(values).any(axis=1)):
        if b - a < window:
            continue
        smooth = savgol_filter(values[a:b], window, 2, axis=0)
        inside = (grid >= t[a]) & (grid <= t[b - 1])
        for k in range(len(columns)):
            out[inside, k] = np.interp(grid[inside], t[a:b], smooth[:, k])
    return grid, out


def find_pushes(grid, box, tip):
    """Cut one clip's 10 Hz tracks into pushes. Returns kept windows and why others were dropped."""
    step = np.linalg.norm(np.diff(box[:, :2], axis=0), axis=1)
    turn = np.abs(wrap(np.diff(box[:, 2]), np.pi / 2))
    moving = np.concatenate([[False], (step > MOVING) | (turn > TURNING)])
    moving = np.nan_to_num(moving).astype(bool)
    tracked = ~np.isnan(box).any(axis=1) & ~np.isnan(tip).any(axis=1)
    gap = np.where(tracked, gap_to_box(np.nan_to_num(tip), np.nan_to_num(box)), np.inf)
    tip_step = np.concatenate([[np.inf], np.linalg.norm(np.diff(tip, axis=0), axis=1)])

    for a, b in runs(~moving):               # a pause of 0.2 s or less doesn't split a push
        if 0 < a and b < len(moving) and b - a <= 2:
            moving[a:b] = True

    kept, dropped = [], {"lost tracking": 0, "moved by hand": 0, "kept gliding": 0, "too short": 0}
    for a, b in runs(moving):
        if b - a < 2:
            dropped["too short"] += 1
            continue
        onset = a - 1                        # last sample before the box moved
        if onset < 0 or not tracked[onset:b].all():
            dropped["lost tracking"] += 1
            continue
        if gap[onset] > 0.03:                # no fingertip at the box: a hand moved it
            dropped["moved by hand"] += 1
            continue
        if (tip_step[b - 2:b] < 0.0005).all():  # box still moving with the finger stopped
            dropped["kept gliding"] += 1
            continue
        # The approach and the stop: only while the finger is down. A finger being lowered or
        # lifted seems to slide sideways (the mark is higher than assumed), so those samples
        # would look like pushes that never happened. Down = still, or moving along the push.
        direction = tip[b - 1] - tip[onset]
        direction /= max(np.linalg.norm(direction), 1e-9)
        start = onset
        while start > max(onset - 10, 1) and tracked[start - 1] and gap[start - 1] < 0.04 \
                and not moving[start - 1] and (tip_step[start] < STILL or
                                               (tip[start] - tip[start - 1]) @ direction > 0.8 * tip_step[start]):
            start -= 1
        end = b
        while end < min(b + 3, len(grid)) and tracked[end] and not moving[end] and tip_step[end] < STILL:
            end += 1
        kept.append((start, onset, end))
    return kept, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks", nargs="+")
    parser.add_argument("--setup", default="data/setup.json", help="for the box's size")
    parser.add_argument("--out", default="data/human_pushes.csv")
    parser.add_argument("--min-nail-area", type=float, default=0.0,
                        help="drop nail detections smaller than this fraction of the clip's typical size (off by default)")
    args = parser.parse_args()
    global HALF
    with open(args.setup) as f:
        HALF = json.load(f)["box_size"] / 2

    episodes, report = [], {}
    for path in args.tracks:
        clip = os.path.splitext(os.path.basename(path))[0]
        track = pd.read_csv(path)
        if "nail_area" in track:             # a partly hidden nail dot gives a biased position
            small = track.nail_area < args.min_nail_area * track.nail_area[track.nail_area > 0].median()
            track.loc[small, ["tip_x", "tip_y"]] = np.nan
        yaw = track.box_yaw.to_numpy()       # unwrap the quarter-turn angle so it moves smoothly
        ok = ~np.isnan(yaw)
        yaw[ok] = yaw[ok][0] + np.concatenate([[0], np.cumsum(wrap(np.diff(yaw[ok]), np.pi / 2))])
        track["box_yaw"] = yaw
        fps = 1 / np.median(np.diff(track.t))
        grid, box = resample(track, ["box_x", "box_y", "box_yaw"], fps)
        _, tip = resample(track, ["tip_x", "tip_y"], fps)
        kept, dropped = find_pushes(grid, box, tip)
        for start, onset, end in kept:
            episodes.append(dict(clip=clip, start=grid[start], onset=onset - start,
                                 box=box[start:end], tip=tip[start:end]))
        report[clip] = dict(pushes=len(kept), dropped=dropped)
        print(f"{clip}: {len(kept)} pushes kept; dropped: " + ", ".join(f"{v} {k}" for k, v in dropped.items() if v))

    # The correction for where the nail dot sits relative to the touching point.
    human_gap = np.median([gap_to_box(e["tip"][e["onset"]][None], e["box"][e["onset"]][None])[0] for e in episodes])
    sim_gap = ROBOT_GAP
    shift = human_gap - sim_gap
    print(f"gap at first contact: finger {1000 * human_gap:.1f} mm, robot ball {1000 * sim_gap:.1f} mm "
          f"-> fingertip moved {1000 * shift:.1f} mm forward along each push")

    rows = []
    for i, e in enumerate(episodes):
        direction = e["tip"][-1] - e["tip"][e["onset"]]
        direction /= max(np.linalg.norm(direction), 1e-9)
        tip = e["tip"] + shift * direction
        for j in range(len(tip)):
            rows.append([i, round(j * STEP, 1), *np.round(tip[j], 5), *np.round(e["box"][j], 5),
                         e["clip"], round(e["start"], 2)])
    out = pd.DataFrame(rows, columns=["episode", "t", "tip_x", "tip_y", "box_x", "box_y", "box_yaw", "clip", "start"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(dict(clips=report, gap_finger_m=human_gap, gap_robot_m=sim_gap, shift_m=shift), f, indent=2)
    print(f"wrote {len(episodes)} pushes, {len(out)} rows, to {args.out}")


if __name__ == "__main__":
    main()
