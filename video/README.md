# Video: from my clips to box and fingertip positions

A phone films the floor from above while I push a box with one finger. These scripts turn each
clip into the box's position and angle and the fingertip's position, in metres on the floor,
then cut the clips into single pushes for the world model. Five steps, in order:

1. calibrate the phone's lens (once);
2. film the pushes;
3. pick the colours of the box, its dots and the finger band (once, or when the light changes);
4. track each clip;
5. cut all the tracks into pushes.

Run everything from the repo's top folder, with `requirements.txt` installed. The videos
themselves (`.mp4`) are not in the repo because of their size: mine are on [Google Drive](https://drive.google.com/drive/folders/1wPU-c3CMfaJyeo-60HNu9z2fGGjO4f7z?usp=sharing);
to replicate, film your own.
My numbers are given at each step, as a reference for what "working" looks like.

## What you need

- A phone that can lock focus and exposure, and something to hold it still, looking straight
  down at the floor (mine lay on a chair seat with the camera sticking out over the edge, about
  47 cm above the floor).
- A plain floor that is darker than white paper (mine: grey concrete).
- A sheet of A4 paper. It is the ruler: its known size (29.7 × 21 cm) gives the camera's
  position and the scale.
- A square box with a strong colour that stands out from the floor (mine: blue plastic,
  12.8 cm square, 5.7 cm tall, measured with a credit card).
- Tape of a second strong colour for two dots on the box (mine: green).
- A band of colour on the pushing finger (mine: a white paper sleeve with a blue band). It can
  be the box's colour, since it is only looked for outside the box, but not the dots' colour.
- A laptop or tablet screen for the calibration board.

## 1. Calibrate the lens (once)

The lens bends straight lines and has its own focal length and centre. Calibration measures
these from many views of a checkerboard, so pixels can later be turned into directions.

**Film.** Show `data/calibration/checkerboard_9x6.png` full screen on a laptop / Ipad (a flat rigid surface), and film it with
the same phone, lens, zoom and resolution as the clips you will film (calibration is only valid for those, with fixed camera position).
Move the screen around for about a minute: across the whole picture including the corners,
near and far, and tilted in different directions by up to about 45°. The board must be sharp
in most frames. My first attempt (small board, barely tilted) gave a focal length 4% off; a
full-screen board, tilted more and reaching the corners, fixed it.

**Run.**
```
python video/calibrate.py data/calibration/calibration.mp4 --views 100 --out data/camera.json
```
It looks at frames spread over the video, finds the board's 9 × 6 inner corners in each, and
uses up to `--views` of them in OpenCV's `calibrateCamera`. The board's real size is not
needed: the scale comes from the A4 sheet later.

**Check.** The printed reprojection error should be below about 1 px (mine: 0.66 px over 100
views). `data/calibration/out/calib_check.png` shows the corners found (left) and the picture
with the lens distortion removed (right): straight lines should look straight.

## 2. Film the pushes

**Phone.** 1080p, 30 fps. Turn off stabilisation and HDR, and lock focus and exposure (on most
phones: press and hold on the floor). Don't zoom. The phone must not move during a clip. It can
move a little between clips, since each clip measures the camera again (mine moved by up to
1 cm and 1.6° between clips).

**Box.** Put the two dots in opposite corners of the box's top, the same distance in from each
corner: their middle is then the box's centre, and the line through them its diagonal. Write
the box's side and height in `data/setup.json` (`box_size`, `box_height`, in metres), and the
finger band's height above the floor while pushing (`nail_height`, about 0.03 for me).

**Each clip.**
- From the very first frame, the A4 sheet lies flat in view, all 4 corners visible, nothing on
  it, for about 3 s. The tracker only looks for it in the first 4 s. Then take it away.
- Push slowly (about 2 cm/s), with the one finger, at different places and angles on all sides.
- Move or turn the box with the other hand every few pushes, to vary where it starts; those
  moments are recognised (no fingertip at the box) and left out.
- Keep the box, its dots and your finger band in view; frames where the hand covers a dot are
  dropped.

I filmed 5 clips of about 4 minutes each (21 minutes in total).

## 3. Pick the colours

The tracker finds things by colour, so it needs the colour ranges of the box, its dots and the
finger band in your light.

```
python video/pick_colours.py data/videos/clip1.mp4 --setup data/setup.json --at 15
```
A window opens at `--at` seconds (choose a moment with the box, both dots and the band in view):
click one of the box's dots, then the finger band, then the box itself, then press any key.
(Without a window: give the three pixels with `--dot x,y --nail x,y --box x,y`.) Each range is
set around the clicked colour, wide enough for the light to vary a little, and saved in
`setup.json`.

**Check** `data/videos/clip1_colours.png`: the box should be painted blue, exactly 2 dots
circled red, and the band circled green. If not, pick again at another moment. Do it again if
the light changes between clips.

## 4. Track each clip

```
python video/track.py data/videos/clip1.mp4 --camera data/camera.json --setup data/setup.json \
    --out data/tracked_clips/clip1.csv --overlay data/tracked_clips/clip1_overlay.mp4
```

**How it works.**
1. *Where the camera is.* In the first 4 s, it looks for the biggest bright four-sided shape at
   a few brightness levels, fits a straight line to each side, and takes the corners where the
   lines cross. From the 4 corners and the sheet's known size, OpenCV's `solvePnP` gives the
   camera's position and tilt. The floor's origin is the sheet's centre.
2. *In every frame, by colour:* the biggest blob of the box's colour is the box; the two biggest
   blobs of the dots' colour on or right next to it are the dots; the biggest blob of the band's
   colour near the box (but not on it) is the finger band.
3. *Pixels to floor positions.* A pixel only gives a direction from the camera. Each mark is
   placed where that direction reaches its own known height: the dots at the box's height, the
   band at `nail_height`. (Treating a dot as lying on the floor would put it up to about 3 cm off.)
4. *The box's pose.* Position = the middle of the two dots. Angle = the direction of the line
   between them, minus 45° (they sit on the diagonal), brought into −45° to 45°, since a square
   looks the same every quarter turn.
5. *Clean-up.* The distance between the two dots can't change, so frames where it is more than
   3 mm off the clip's median (a dot partly hidden by the hand) are dropped.

**Check.**
- The printed A4 fit should be under about 2 px, and the camera height plausible (mine: 0.7 to
  1.8 px, 46.6 to 47.6 cm).
- `<out>_a4.png`: the red outline should sit exactly on the sheet.
- The printed share of frames with the box found (mine: 74 to 94%) and the dot spacing's spread
  (mine: 0.6 to 1.5 mm, which is the tracking noise).
- The overlay video: the yellow square (box top) and orange square (box bottom) should sit on
  the box, and the green circle on the band. The squares are drawn from the computed pose and
  the box's known size, not fitted to the picture, so if they sit right, the tracking is right.

**Options.**
- `--overlay`: also write a checking video (half size) with the tracking drawn on.
- `--a4-corners x1,y1,...,x4,y4`: the sheet's corners in pixels, if it isn't found by itself.

**Writes.**
- `<out>.csv`: one row per frame: `frame, t, box_x, box_y, box_yaw, dot_gap, tip_x, tip_y,
  nail_area` (metres and radians, on the floor). Blank where something wasn't found.
  `nail_area` is the band's size in pixels: when it shrinks, the band is partly hidden.
- `<out>.json`: the camera's position, how well the A4 corners fit, and the measured dot spacing.
- `<out>_a4.png`: the check image above.

## 5. Cut the tracks into pushes

```
python video/build_dataset.py data/tracked_clips/clip[1-5].csv --setup data/setup.json \
    --min-nail-area 0.3 --out data/human_pushes.csv
```

**How it works.**
1. *Smooth and resample.* Each track is smoothed (Savitzky–Golay, about 0.2 s) and sampled
   every 0.1 s: long enough that the box moves more per step (about 2 mm at 2 cm/s) than
   the tracking noise (about 1 mm), short enough that a push keeps 10 to 20 steps. Gaps of up to 3 frames are filled in; longer ones
   break the track. Band detections smaller than 0.3 of the clip's typical size are dropped
   first (`--min-nail-area`), because a partly hidden band gives a biased position.
2. *Find pushes.* A push is a stretch where the box moves (more than 0.8 mm or 0.4° per 0.1 s)
   while the fingertip is right next to it. Up to 1 s before (the approach) and 0.3 s after (the
   stop) are kept, but only while the finger is down.
3. *Drop the bad ones:* tracking lost during the push, no fingertip within 3 cm of the box when
   it started moving (moved by hand), the box still gliding after the finger stopped, or too short.
4. *Line up finger and robot.* The tracker follows the band's centre, which sits behind the point
   that touches the box. When the box starts moving, the band's centre is typically 17.7 mm from
   the box's side (mine); the robot's ball centre is 8 mm from it (its radius). So every fingertip
   position is moved forward along its push by the difference (9.7 mm for me), and "where the
   pusher is" means the same point for both.

**Check.** It prints, per clip, the pushes kept and why others were dropped, then the finger gap
and the shift. Mine: 150 pushes (30, 29, 23, 25 and 43 per clip); about half of the candidate
pushes were dropped for lost tracking. The same report is saved in `data/human_pushes.json`.

**Writes** `data/human_pushes.csv`: `episode, t, tip_x, tip_y, box_x, box_y, box_yaw, clip, start`
(one episode = one push; `start` is when it begins in its clip, in seconds).

## Files

| File | What it does |
|---|---|
| `calibrate.py` | Step 1: the lens, from the checkerboard video |
| `pick_colours.py` | Step 3: the colour ranges, from three clicks |
| `track.py` | Step 4: the camera from the A4 sheet, then the box and fingertip in every frame |
| `build_dataset.py` | Step 5: the pushes |
| `geometry.py` | The camera maths: the A4 sheet to the camera's pose, pixel to floor position at a given height |

## `data/setup.json`

| Key | Meaning |
|---|---|
| `box_size` | the box's side (m) |
| `box_height` | the box's height (m): the dots are found at this height |
| `nail_height` | the finger band's height above the floor while pushing (m). ("nail": my first finger mark was a dot on the nail) |
| `colours` | colour ranges for `blue` (the box), `dots` (the two dots on its top) and `nail` (the finger band), set by `pick_colours.py` |

## When things go wrong (from my runs)

- **The A4 sheet isn't found, or the camera height looks wrong**: the sheet may have merged with
  a light patch of floor. Check `<out>_a4.png`; use a darker, plainer floor, or give the corners
  with `--a4-corners`.
- **The box is found in few frames**: the dots' colour is too close to the floor's or the box's.
  A white strip on my light grey floor was found in only 25 to 35% of frames; green dots in 74 to 94%.
- **The fingertip appears where there is no finger**: specks of the band's colour on the box's
  edge. They are tiny, and `--min-nail-area` drops them.
- **Few pushes kept**: most are lost to tracking gaps. Keep the dots and band in view, and push
  slower.
