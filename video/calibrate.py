"""Code to measure the camera and focal settings of our workspace.

Run:  python video/calibrate.py calib.mp4 --out data/camera.json

It looks for the board's 9 x 6 inner corners in frames spread over the video, then
OpenCV's calibrateCamera finds the focal length, image centre and lens distortion that
best explain where those corners appear. The board's real size isn't needed for this.

Check: the printed error should be below about 1 pixel, and out/calib_check.png (next to the video) should
show coloured corner markers sitting exactly on the board's corners.
"""
import argparse
import json
import os

import cv2
import numpy as np

PATTERN = (9, 6)  # inner corners of checkerboard_9x6.png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out", default="data/camera.json")
    parser.add_argument("--views", type=int, default=40, help="how many board views to use at most")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    board = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)  # in board squares

    found, size, example = [], None, None
    for i in np.linspace(0, n_frames - 1, min(n_frames, 4 * args.views)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            found.append(corners.astype(np.float32))
            if example is None:
                example = (frame, corners)
    print(f"board found in {len(found)} of the frames looked at")
    if len(found) < 8:
        raise SystemExit("too few board views: film the tablet from more angles, and check it's in focus")

    found = [found[i] for i in np.linspace(0, len(found) - 1, min(len(found), args.views)).astype(int)]
    rms, K, dist, _, _ = cv2.calibrateCamera([board] * len(found), found, size, None, None)
    print(f"reprojection error {rms:.2f} px (good if below ~1)")
    print(f"focal length {K[0, 0]:.0f} px, image centre ({K[0, 2]:.0f}, {K[1, 2]:.0f}), image {size[0]} x {size[1]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(K=K.tolist(), dist=dist.ravel().tolist(), image_size=list(size),
                       rms_px=rms, views=len(found)), f, indent=2)
    check = os.path.join(os.path.dirname(args.video), "out", "calib_check.png")  # next to the video
    os.makedirs(os.path.dirname(check), exist_ok=True)
    frame, corners = example
    cv2.drawChessboardCorners(frame, PATTERN, corners, True)
    cv2.imwrite(check, np.hstack([frame, cv2.undistort(frame, K, dist)]))
    print(f"wrote {args.out} and {check} (left: corners found; right: lens distortion removed)")


if __name__ == "__main__":
    main()
