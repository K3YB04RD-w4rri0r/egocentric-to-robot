"""Simple tracker, I can directly track the colors since the lighting and objects are the same. 
Done by clicking on them.

Run:  python video/pick_colours.py clip.mp4 --setup data/setup.json --at 20
      A window opens at 10 s into the clip: click one of the box's dots, then the band
      on your finger, then the box itself, then press any key.
Or, without a window, give the pixels (x,y) yourself:
      python video/pick_colours.py clip.mp4 --setup data/setup.json --at 10 --dot 812,404 --nail 700,455 --box 790,430

Each colour range is set around the clicked colour, wide enough for the light to vary a bit
over the clip, so yellow or dim light doesn't matter. The ranges are saved in the setup file,
and <clip>_colours.png shows what each one picks up in that frame, to check.
"""
import argparse
import json
import os

import cv2
import numpy as np

from track import colour_mask, find_markers

NAMES = ("dot", "nail", "box")
LABELS = ("box's dot", "finger mark", "box itself")
KEYS = dict(dot="dots", nail="nail", box="blue")


def grab(path, seconds):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * seconds)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"could not read {path} at {seconds} s")
    return frame


def click(frame):
    """Let the user click the three things in turn; returns their pixels."""
    points, shown = [], frame.copy()

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x, y))
            cv2.circle(shown, (x, y), 6, (0, 0, 255), 2)

    cv2.namedWindow("pick", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pick", on_mouse)
    while True:
        prompt = f"click the {LABELS[len(points)]}" if len(points) < 3 else "done: press any key"
        view = shown.copy()
        cv2.putText(view, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imshow("pick", view)
        if cv2.waitKey(30) != -1 and len(points) == 3:
            break
    cv2.destroyAllWindows()
    return dict(zip(NAMES, points))


def colour_range(hsv, xy, r=3):
    """An HSV range around the median colour of the pixels near xy."""
    x, y = xy
    patch = hsv[y - r:y + r + 1, x - r:x + r + 1].reshape(-1, 3).astype(int)
    h, s, v = np.median(patch, axis=0).astype(int)
    if s < 30:   # whitish or grey: its hue means nothing, it is just pale and bright
        return dict(h=[0, 179], s=[0, int(min(255, s + 50))], v=[int(max(0, v - 30)), 255])
    return dict(h=[int((h - 12) % 180), int((h + 12) % 180)],   # may wrap round red; colour_mask handles it
                s=[int(s // 2), 255], v=[int(max(0, v - 80)), 255])   # greys below half its saturation have no real hue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("--setup", required=True)
    parser.add_argument("--at", type=float, default=10.0, help="seconds into the clip")
    for name in NAMES:
        parser.add_argument(f"--{name}", help="pixel x,y (skips the window)")
    args = parser.parse_args()

    frame = grab(args.clip, args.at)
    if all(getattr(args, n) for n in NAMES):
        points = {n: tuple(int(v) for v in getattr(args, n).split(",")) for n in NAMES}
    else:
        points = click(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    with open(args.setup) as f:
        setup = json.load(f)
    for name, xy in points.items():
        setup["colours"][KEYS[name]] = colour_range(hsv, xy)
        print(f"{name}: clicked colour range {setup['colours'][KEYS[name]]}")

    # Check: what the tracker now picks up in this frame.
    dots, nail, _ = find_markers(frame, setup["colours"])
    view = frame.copy()
    view[colour_mask(hsv, setup["colours"]["blue"])] = (255, 120, 0)
    for d in dots:
        cv2.circle(view, tuple(int(v) for v in d), 9, (0, 0, 255), 2)
    if nail is not None:
        cv2.circle(view, tuple(int(v) for v in nail), 9, (0, 255, 0), 2)
    out = os.path.splitext(args.clip)[0] + "_colours.png"
    cv2.imwrite(out, np.hstack([frame, view]))
    print(f"in this frame: {len(dots)} dots on the box (2 expected), "
          f"finger mark {'found' if nail is not None else 'NOT found'}")
    print(f"check {out}: box painted blue, box dots circled red, nail circled green")

    with open(args.setup, "w") as f:
        json.dump(setup, f, indent=2)
    print(f"saved the colours in {args.setup}")


if __name__ == "__main__":
    main()
