"""Track the box and your fingertip in one clip, frame by frame.

Run:  python video/track.py clip.mp4 --camera data/camera.json --setup data/setup.json \
          --out data/tracked_clips/clip01.csv

First, the A4 sheet in the first seconds tells where the camera is. Then, in every frame:
1. find the blue box, the two green dots on its top, and the blue band on your finger;
2. turn each dot's pixel into a position on the floor, using the dot's own height;
3. box position = midpoint of its two dots; box angle = direction from one dot to the
   other minus 45° (the dots sit on the box's diagonal),
   modulo 90° (a square repeats every quarter turn).

Output, one row per video frame (metres and radians, in the A4 sheet's frame):
  frame, t, box_x, box_y, box_yaw, dot_gap, tip_x, tip_y, nail_area
Cells are blank when something wasn't found. dot_gap is the measured distance between the two
dots: a built-in accuracy check, since it should never change.
nail_area is the nail dot's size in pixels: when it shrinks, the dot is partly hidden.
Also writes <out>.json (camera position, A4 fit), <out>_a4.png (the sheet's outline as the
fitted camera sees it: it should sit exactly on the sheet) and, with --overlay, a video
with the box outline and fingertip drawn on, to check by eye.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

from geometry import a4_corners_3d, back_project, box_corners_3d, camera_pose_from_a4, load_camera, project

KERNEL = np.ones((3, 3), np.uint8)


def colour_mask(hsv, c):
    """Pixels inside an HSV colour range. Hue runs 0-179 in OpenCV; h = [170, 10] wraps round red."""
    (h0, h1), (s0, s1), (v0, v1) = c["h"], c["s"], c["v"]
    if h0 <= h1:
        return cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1)) > 0
    return (cv2.inRange(hsv, (h0, s0, v0), (179, s1, v1)) | cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))) > 0


def blobs(mask, min_area, keep):
    """Centres (sub-pixel) and areas of the `keep` largest blobs in a mask, largest first."""
    n, _, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    order = [i for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1 if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return centres[order[:keep]], stats[order[:keep], cv2.CC_STAT_AREA]


def find_markers(frame, colours):
    """Pixel positions of the box's dot candidates, and the finger band's position and area."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.morphologyEx(colour_mask(hsv, colours["blue"]).astype(np.uint8), cv2.MORPH_OPEN, KERNEL)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    if n < 2:
        return np.empty((0, 2)), None, 0
    box = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])     # the biggest blue blob is the box
    hull = np.zeros(box.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(box.astype(np.uint8))), 1)
    # The dots sit in the corners and hide the box's colour there, so look a little beyond the
    # box's outline (a tenth of its width) to see them whole.
    margin = int(0.1 * np.sqrt(box.sum())) | 1
    light = colour_mask(hsv, colours["dots"]) & (cv2.dilate(hull, np.ones((margin, margin), np.uint8)) > 0) & ~box
    dots = blobs(light, min_area=4, keep=5)[0]

    # The nail is searched for only near the box, so green things elsewhere don't matter.
    reach = int(2 * np.sqrt(box.sum()))
    near = cv2.dilate(hull, np.ones((reach, reach), np.uint8)) > 0
    nail, area = blobs(colour_mask(hsv, colours["nail"]) & near & ~box, min_area=4, keep=1)
    return dots, (nail[0] if len(nail) else None), (int(area[0]) if len(area) else 0)


def box_from_dots(candidates, cam, setup):
    """The box's pose from its two biggest dots.

    Returns (x, y, yaw, measured gap, the two marks' pixels) or None. A partly hidden mark
    changes the gap; such frames are dropped after the clip, in main.
    """
    if len(candidates) < 2:
        return None
    pts = back_project(candidates[:2], *cam, setup["box_height"])
    a, b, gap, pair_px = pts[0], pts[1], np.linalg.norm(pts[0] - pts[1]), candidates[[0, 1]]
    centre = (a + b) / 2
    # The two marks sit on the box's diagonal, so its sides are 45° from the line through them.
    edge = np.arctan2(b[1] - a[1], b[0] - a[0]) - np.pi / 4
    yaw = (edge + np.pi / 4) % (np.pi / 2) - np.pi / 4        # same square, angle brought into [-45°, 45°)
    return centre[0], centre[1], yaw, gap, pair_px


def find_a4(cap, K, dist, seconds=4.0):
    """Find the A4 sheet in the first seconds and return the camera pose it gives.

    In each frame looked at, the biggest bright four-sided shapes are tried, at a few
    brightness levels; each side is fitted with a straight line through its edge pixels, and
    the corners are where the lines cross. The shape whose corners fit an A4 sheet best wins.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    best = None
    for i in range(0, int(seconds * fps), 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # The automatic (Otsu) level can fall between a lighter and a darker patch of floor,
        # merging the sheet with the lighter patch, so brighter levels are tried too.
        for level in range(int(otsu), 240, 10):
            bright = (gray > level).astype(np.uint8)
            contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
                if cv2.contourArea(c) < 0.01 * gray.size:
                    break
                hull = cv2.convexHull(c)  # fills in any notch where something overlaps the sheet's edge
                quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
                if len(quad) != 4:
                    continue
                corners = refine_corners(c.reshape(-1, 2), quad.reshape(4, 2), K, dist)
                if corners is None:
                    continue
                try:
                    R, t, err = camera_pose_from_a4(corners, K, np.zeros(5))
                except ValueError:
                    continue
                if best is None or err < best[2]:
                    best = (R, t, err, i, corners, frame)
    if best is None:
        raise SystemExit("couldn't find the A4 sheet in the first seconds; is it flat and fully in view?")
    return best


def refine_corners(contour, quad, K, dist):
    """Corners as crossings of straight lines fitted to each side (lens distortion removed).

    Returns None if a side has too few edge pixels to be a real straight edge."""
    pts = cv2.undistortPoints(contour.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(-1, 2)
    quad = cv2.undistortPoints(quad.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(4, 2)
    lines = []
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        ab = b - a
        along = (pts - a) @ ab / (ab @ ab)                      # 0 at a, 1 at b
        across = np.abs(np.cross(ab, pts - a)) / np.linalg.norm(ab)
        side = pts[(along > 0.1) & (along < 0.9) & (across < 6)]  # edge pixels away from corners
        if len(side) < 10:
            return None  # this side isn't a real straight edge
        vx, vy, x0, y0 = cv2.fitLine(side.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append(np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0]))  # line in homogeneous form
    corners = [np.cross(lines[k - 1], lines[k]) for k in range(4)]
    return np.array([c[:2] / c[2] for c in corners])


def draw(frame, cam, setup, box, tip_px, nail_xy):
    """Box outline (top bright, bottom dim), box dots and fingertip, drawn on a frame."""
    if box is not None:
        x, y, yaw, _, dots_px = box
        for z, colour in ((0.0, (0, 140, 255)), (setup["box_height"], (0, 255, 255))):
            px = project(box_corners_3d(x, y, yaw, setup["box_size"], z), *cam).astype(int)
            cv2.polylines(frame, [px], True, colour, 2)
        for p in dots_px:
            cv2.circle(frame, tuple(np.round(p).astype(int)), 6, (255, 0, 255), 2)
    if tip_px is not None:
        cv2.circle(frame, tuple(np.round(tip_px).astype(int)), 9, (0, 255, 0), 2)
        floor = project([[*nail_xy, 0.0]], *cam)[0]             # the fingertip's spot on the floor
        cv2.drawMarker(frame, tuple(np.round(floor).astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--camera", default="data/camera.json")
    parser.add_argument("--setup", default="data/setup.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overlay", help="also write a checking video here (.mp4)")
    parser.add_argument("--a4-corners", help="if the sheet isn't found: its 4 corners in the first frame, "
                                             "in order round the sheet, as x1,y1,x2,y2,x3,y3,x4,y4 (pixels)")
    args = parser.parse_args()

    with open(args.setup) as f:
        setup = json.load(f)
    K, dist = load_camera(args.camera)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"can't open the video {args.video}: check the path")
    if args.a4_corners:
        a4_px = np.array([float(v) for v in args.a4_corners.split(",")]).reshape(4, 2)
        a4_px = cv2.undistortPoints(a4_px.reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2)
        R, t, err = camera_pose_from_a4(a4_px, K, np.zeros(5))
        a4_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        a4_image = cap.read()[1]
    else:
        R, t, err, a4_frame, a4_px, a4_image = find_a4(cap, K, dist)
    print(f"A4 sheet found in frame {a4_frame}, corners fit to {err:.1f} px "
          f"({'good' if err < 2 else 'POOR: check the picture'}); camera {100 * (-R.T @ t)[2]:.0f} cm above the floor")
    cam = (K, dist, R, t)
    # Check picture: the sheet's outline as the fitted camera sees it, and the floor axes.
    outline = project(np.vstack([a4_corners_3d(True), a4_corners_3d(True)[:1]]), *cam).astype(int)
    cv2.polylines(a4_image, [outline], False, (0, 0, 255), 2)
    origin, x_end, y_end = project([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], *cam).astype(int)
    cv2.arrowedLine(a4_image, tuple(origin), tuple(x_end), (0, 0, 255), 3)
    cv2.arrowedLine(a4_image, tuple(origin), tuple(y_end), (0, 200, 0), 3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(os.path.splitext(args.out)[0] + "_a4.png", a4_image)

    info = dict(R=R.tolist(), t=t.tolist(), a4_fit_px=err, a4_frame=a4_frame, a4_corners_px=a4_px.tolist())
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(info, f, indent=2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    rows, found_box, found_tip, i = [], 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 or i / fps
        dots_px, nail_px, nail_area = find_markers(frame, setup["colours"])
        box = box_from_dots(dots_px, cam, setup)
        tip = back_project([nail_px], *cam, setup["nail_height"])[0] if nail_px is not None else None
        found_box += box is not None
        found_tip += tip is not None
        rows.append([i, round(t_s, 4),
                     *(np.round(box[:4], 5) if box is not None else [""] * 4),
                     *(np.round(tip, 5) if tip is not None else [""] * 2), nail_area])
        if args.overlay:
            frame = draw(frame, cam, setup, box, nail_px, tip)
            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            if writer is None:
                writer = cv2.VideoWriter(args.overlay, cv2.VideoWriter_fourcc(*"mp4v"), fps, small.shape[1::-1])
            writer.write(small)
        i += 1

    # A hand covering part of a mark makes the gap between the two marks look shorter (or
    # longer) and moves their middle: drop the frames whose gap is off by more than 3 mm.
    lengths = np.array([r[5] if r[5] != "" else np.nan for r in rows], float)
    off = np.abs(lengths - np.nanmedian(lengths)) > 0.003
    for r in np.flatnonzero(off):
        rows[r][2:6] = [""] * 4
    found_box -= int(off.sum())

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "box_x", "box_y", "box_yaw", "dot_gap", "tip_x", "tip_y", "nail_area"])
        w.writerows(rows)
    gaps = np.array([r[5] for r in rows if r[5] != ""], float)
    print(f"{i} frames: box found in {100 * found_box / max(i, 1):.0f}%, fingertip in {100 * found_tip / max(i, 1):.0f}%")
    if len(gaps):
        print(f"box dot spacing measured "
              f"{1000 * np.median(gaps):.1f} mm, spread {1000 * np.std(gaps):.1f} mm")
        info["measured_gap_m"] = round(float(np.median(gaps)), 5)
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(info, f, indent=2)
    if writer:
        writer.release()
    print(f"wrote {args.out}" + (f" and {args.overlay}" if args.overlay else ""))


if __name__ == "__main__":
    main()
