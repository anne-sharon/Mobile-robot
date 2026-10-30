"""Screenshots of one docking run for the project page (evaluation only).
Run from the repo root after `python -m eval.make_video nominal`:  python -m eval.episode_screenshots
Reads eval/results/docking_nominal.mp4 and writes
  media/episode_stages.png : the overhead view at four planner stages (2 x 2)
  media/episode_parked.png : the final frame (parked, camera view, estimate error with its 2-sigma band)
"""
import sys
import cv2
import numpy as np
import imageio.v2 as imageio

# frame index -> label (frames are 0.1 s of simulation apart: index 80 = t 8 s)
STAGES = [(80, "1  Search: spin on the spot to look for the tag"),
          (140, "2  Explore: the red box blocks the view"),
          (380, "3  Approach: the LiDAR steers between boxes"),
          (530, "4  Dock: follow the dock's centre line in")]
TOP_W, TOP_H, DRAWN_H = 720, 480, 690       # overhead size, and the frame height make_video drew


def screenshots(video="eval/results/docking_nominal.mp4", out_dir="media"):
    r = imageio.get_reader(video)
    n = r.count_frames()
    scale = r.get_data(0).shape[0] / DRAWN_H   # the encoder pads the height to a multiple of 8
    tiles = []
    for i, label in STAGES:
        f = r.get_data(min(i, n - 1))
        top = cv2.resize(f[0:int(round(TOP_H * scale)), 0:TOP_W], (TOP_W, TOP_H))
        cv2.rectangle(top, (0, 0), (TOP_W, 40), (46, 8, 18), -1)
        cv2.putText(top, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (246, 238, 233), 2, cv2.LINE_AA)
        tiles.append(top)
    gap_v = np.full((TOP_H, 12, 3), 255, np.uint8)
    rows = [np.hstack([tiles[0], gap_v, tiles[1]]), np.hstack([tiles[2], gap_v, tiles[3]])]
    grid = np.vstack([rows[0], np.full((12, rows[0].shape[1], 3), 255, np.uint8), rows[1]])
    cv2.imwrite(f"{out_dir}/episode_stages.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(f"{out_dir}/episode_parked.png", cv2.cvtColor(r.get_data(n - 1), cv2.COLOR_RGB2BGR))
    return n


if __name__ == "__main__":
    print(screenshots(*sys.argv[1:]))
