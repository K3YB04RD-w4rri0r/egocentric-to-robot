"""Push the box to a target position and angle with the simulated robot.

Loop, at most 15 pushes:
  1. read where the box is from the simulator;
  2. make up 500 random pushes (which face, where on it, at what angle, how far);
  3. ask the world model where each would leave the box, and score it: mm and degrees off
     the target, each divided by the tolerance, averaged over the 5 networks;
  4. do the best-scoring push with the robot;
  5. stop once the box is within the tolerance (default 1.5 cm and 10 degrees).
The box is square, so an angle 90 degrees off looks the same and counts as reached.

Run:  MUJOCO_GL=egl python plan/planner.py --policy model --model models/video.pt --trials 30 \
          --out results/plan_video.json

Options:
  --policy model        the loop above, with the world model given by --model
  --policy rule         a hand-written rule: push near a corner if the angle is off, else push
                        straight at the target through the middle
  --policy rule-offset  the same rule, but it turns the box less by hitting closer to the middle
                        instead of by pushing less far (for --length, where it can't choose how far)
  --within MM DEG       the tolerance (default 15 10)
  --length 0.03         every push exactly 3 cm: only where and at what angle is left to choose
  --ahead 2             the model looks two pushes ahead (100 first pushes x 100 follow-ups each)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "wm"))
from model import load_model, rollout  # noqa: E402
from push_env import BOX_SIZE, GRIPPER_CLOSED, make_env, push, push_path  # noqa: E402

CANDIDATES = 500
MAX_PUSHES = 15
SPEED = 0.03                              # m/s, the robot's pushing speed
POS_OK, ANGLE_OK = 0.015, np.radians(10)  # success: within 1.5 cm and 10 degrees
START_AREA = dict(x=(-0.03, 0.05), y=(-0.07, 0.07))  # where starts and targets are drawn
FIXED_LENGTH = None                       # metres: every push this long (--length), else each push chooses
AHEAD = 1                                 # pushes the model looks ahead (--ahead)
AHEAD_CANDIDATES = 100                    # with --ahead 2: first pushes, and follow-ups to each


def angle_gap(a, b):
    """How far angle a is from angle b for a square box (it repeats every 90 degrees)."""
    return np.abs((a - b + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def cost(pose, target):
    """Distance to the target in units of the success thresholds (below ~1 is close)."""
    pos = np.linalg.norm(pose[..., :2] - target[:2], axis=-1)
    return pos / POS_OK + angle_gap(pose[..., 2], target[2]) / ANGLE_OK


def random_pushes(rng, n):
    return dict(side=rng.integers(4, size=n), offset=rng.uniform(-1, 1, n),
                angle=rng.uniform(-np.pi / 4, np.pi / 4, n),
                length=np.full(n, FIXED_LENGTH) if FIXED_LENGTH else rng.uniform(0.01, 0.035, n))


def tip_path(box, side, offset, angle, length, steps):
    """Where the pusher will be every 0.1 s during a push (straight line at SPEED, then still)."""
    start, end = push_path(box, side, offset, angle, length)
    travelled = np.minimum(SPEED * 0.1 * np.arange(steps), np.linalg.norm(end - start))
    direction = (end - start) / np.linalg.norm(end - start)
    return start + travelled[:, None] * direction


def predict(model, boxes, c, box0=None):
    """Where the world model says each push in c leaves the box: members x N x 3. boxes: N x 3 poses the pushes
    are aimed from; box0 (members x N x 3): carry on from each network's own earlier prediction instead."""
    steps = int(np.ceil((0.02 + 0.035) / (SPEED * 0.1))) + 3   # long enough for the longest push, then a stop
    tips = np.stack([tip_path(boxes[i], c["side"][i], c["offset"][i], c["angle"][i], c["length"][i], steps)
                     for i in range(len(boxes))])
    return rollout(model, boxes if box0 is None else box0, tips).numpy()[:, :, -1]


def model_push(model, box, target, rng):
    """The push, out of CANDIDATES random ones, that the world model says lands closest to the target."""
    c = random_pushes(rng, CANDIDATES)
    scores = cost(predict(model, np.repeat(box[None], CANDIDATES, 0), c), target).mean(0)
    best = int(np.argmin(scores))
    return {k: float(v[best]) for k, v in c.items()}, float(scores[best])


def model_push_ahead(model, box, target, rng, n=AHEAD_CANDIDATES):
    """Looking two pushes ahead: n random first pushes, each followed by n random second pushes from where the
    model says the first leaves the box. A first push scores as well as its best follow-up, or as itself if
    that is better. Only the first push is done; the next one is planned again from where the box really is."""
    c = random_pushes(rng, n)
    first = predict(model, np.repeat(box[None], n, 0), c)                        # members x n x 3
    second = predict(model, np.repeat(first.mean(0), n, 0), random_pushes(rng, n * n),
                      box0=np.repeat(first, n, 1))                               # members x n*n x 3
    scores = np.minimum(cost(first, target).mean(0), cost(second, target).mean(0).reshape(n, n).min(1))
    best = int(np.argmin(scores))
    return {k: float(v[best]) for k, v in c.items()}, float(scores[best])


def rule_push(box, target, by_offset=False):
    """Hand-written: fix the angle with a corner push, then push straight at the target.
    by_offset: set how much it turns by where it hits (closer to the middle turns less), not by how far."""
    to_target = target[:2] - box[:2]
    turn = (target[2] - box[2] + np.pi / 4) % (np.pi / 2) - np.pi / 4
    # Pick the face whose straight-in push points most toward the target.
    sides = np.arange(4)
    inward = -np.stack([np.cos(box[2] + sides * np.pi / 2), np.sin(box[2] + sides * np.pi / 2)], 1)
    side = int(np.argmax(inward @ to_target))
    if abs(turn) > ANGLE_OK * 0.7:
        # A push near a corner turns the box about 9 degrees per 3 cm; positive offset turns it counter-clockwise.
        if by_offset:
            length = FIXED_LENGTH or 0.03
            offset = np.sign(turn) * min(1.0, abs(turn) / (np.radians(9) * length / 0.03))
            return dict(side=side, offset=float(offset), angle=0.0, length=float(length)), None
        length = FIXED_LENGTH or np.clip(abs(turn) / np.radians(9) * 0.03, 0.01, 0.035)
        return dict(side=side, offset=float(np.sign(turn)), angle=0.0, length=float(length)), None
    a, b = inward[side], to_target
    angle = np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)   # from straight-in to the target direction
    return dict(side=side, offset=0.0, angle=float(np.clip(angle, -np.pi / 4, np.pi / 4)),
                length=float(FIXED_LENGTH or np.clip(np.linalg.norm(to_target), 0.01, 0.035))), None


def random_task(rng):
    """A start pose and a target pose 3 to 8 cm and up to 45 degrees away."""
    while True:
        start = np.array([rng.uniform(*START_AREA["x"]), rng.uniform(*START_AREA["y"]), rng.uniform(-np.pi, np.pi)])
        d, heading = rng.uniform(0.03, 0.08), rng.uniform(-np.pi, np.pi)
        target = start + [d * np.cos(heading), d * np.sin(heading), rng.uniform(-np.pi / 4, np.pi / 4)]
        if START_AREA["x"][0] <= target[0] <= START_AREA["x"][1] and START_AREA["y"][0] <= target[1] <= START_AREA["y"][1]:
            return start, target


def run_task(env, policy, start, target, rng, model=None, on_step=None):
    env.set_box_pose(*start)
    for _ in range(5):
        env.step(np.array([0, 0, 0, GRIPPER_CLOSED]))
    history = [env.box_pose().tolist()]
    for n in range(MAX_PUSHES):
        box = env.box_pose()
        if cost_parts(box, target)[2]:
            break
        if policy == "model":
            p, _ = (model_push_ahead if AHEAD == 2 else model_push)(model, box, target, rng)
        else:
            p, _ = rule_push(box, target, policy == "rule-offset")
        push(env, p["side"], p["offset"], p["angle"], p["length"], speed=SPEED, on_step=on_step)
        history.append(env.box_pose().tolist())
    pos, ang, ok = cost_parts(env.box_pose(), target)
    return dict(start=start.tolist(), target=target.tolist(), poses=history, pushes=len(history) - 1,
                success=bool(ok), pos_err_mm=1000 * pos, angle_err_deg=float(np.degrees(ang)))


def cost_parts(pose, target):
    pos = float(np.linalg.norm(pose[:2] - target[:2]))
    ang = float(angle_gap(pose[2], target[2]))
    return pos, ang, pos < POS_OK and ang < ANGLE_OK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["model", "rule", "rule-offset"], default="model")
    parser.add_argument("--model")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--within", type=float, nargs=2, default=(15, 10), metavar=("MM", "DEG"),
                        help="success: this close to the target (default 15 mm and 10 degrees)")
    parser.add_argument("--length", type=float, help="every push exactly this long (metres)")
    parser.add_argument("--ahead", type=int, choices=[1, 2], default=1, help="pushes the model looks ahead")
    args = parser.parse_args()
    torch.set_num_threads(1)
    global POS_OK, ANGLE_OK, FIXED_LENGTH, AHEAD
    POS_OK, ANGLE_OK = args.within[0] / 1000, np.radians(args.within[1])
    FIXED_LENGTH = args.length
    AHEAD = args.ahead

    model = load_model(args.model)[0] if args.policy == "model" else None
    env = make_env()
    tasks = np.random.default_rng(1000 + args.seed)       # the same tasks for every policy
    rng = np.random.default_rng(args.seed)
    results, t0 = [], time.time()
    for i in range(args.trials):
        start, target = random_task(tasks)
        r = run_task(env, args.policy, start, target, rng, model)
        results.append(r)
        print(f"task {i + 1}: {'reached' if r['success'] else 'missed '} after {r['pushes']} pushes "
              f"({r['pos_err_mm']:.0f} mm, {r['angle_err_deg']:.0f} deg off)", flush=True)
    won = [r for r in results if r["success"]]
    print(f"{args.policy}: reached {len(won)}/{len(results)} targets"
          + (f", median {np.median([r['pushes'] for r in won]):.0f} pushes when reached" if won else "")
          + f" ({time.time() - t0:.0f} s)")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(policy=args.policy, model=args.model, within_mm_deg=args.within, length=args.length,
                           ahead=args.ahead, results=results), f, indent=1)


if __name__ == "__main__":
    main()
