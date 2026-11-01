"""The default docking run with a caption explaining every movement (for learning and the demo video).
Run from the repo root:  python -m eval.explained_video
Writes media/docking_explained.mp4 at half the speed of the other videos (1.5x real time).

The times below come from replaying the default run (start (-2.3, -1.2), facing 138 deg) and logging the planner's
stage, its avoidance decisions and the robot's motion. The run is deterministic, so they match the clock shown in
the video's top bar. If the robot code changes, replay the run and update them.
"""
from eval.make_video import make_video

CAPTIONS = [
    (0.0, 9.4, "SEARCH: spins once on the spot",
     "It doesn't know where the dock is. From the start, the red box blocks the line of sight to the tag."),
    (9.4, 16.9, "EXPLORE: turns north, drives about 1.3 m up the left wall",
     "The LiDAR picks the most open, not-yet-visited spot (max 1.5 m). Spinning again here would see the same thing."),
    (16.9, 23.4, "SEARCH AGAIN: spins about 250 deg",
     "The tag appears at the edge of the camera's view, 5.4 m away. The EKF switches to the dock frame."),
    (23.4, 24.3, "APPROACH: drives off angled up-left",
     "The first fix from 5.4 m is rough (about 1 m off sideways): a few degrees of angle error at 5 m is a lot."),
    (24.3, 27.7, "APPROACH: stops and turns on the spot back to about -9 deg",
     "Each frame improves the estimate, so the target moves. More than 46 deg off -> turn on the spot first."),
    (27.7, 28.4, "APPROACH: drives east, curving slightly",
     "The LQR steers back onto the line to the point 0.8 m in front of the dock."),
    (28.4, 30.1, "AVOID: small swerve right past the coffee table",
     "The table's edge is inside the 0.4 m corridor checked in the LiDAR scan, so it takes the nearest clear direction."),
    (30.1, 32.6, "APPROACH: drives straight east",
     "The corridor ahead is clear, so it heads straight for the approach point."),
    (32.6, 39.2, "AVOID: edges right, past the blue box",
     "The box's lower edge is about 0.14 m from the path. It re-plans every step, 0.7 m ahead, with no map."),
    (39.2, 43.2, "AVOID: swings left between the blue and green boxes",
     "The green box blocks the right, so the first clear direction is left. It stays about 0.4 m from both."),
    (43.2, 48.1, "APPROACH: final leg to 0.8 m in front of the tag",
     "The obstacles are behind it. It arrives facing the tag, so the align stage has nothing to do."),
    (48.1, 55.9, "DOCK: creeps in along the centre line, 12 -> 2 cm/s",
     "Speed = 0.6 x distance left, so it stops smoothly. Up close the estimate is within millimetres."),
    (55.9, 999.0, "PARKED: 0.35 m from the tag",
     "Already within 1 deg of square, so no square-up turn was needed. 0.7 cm and 0.5 deg from the target."),
]

if __name__ == "__main__":
    print(make_video("nominal", out="media/docking_explained.mp4", fps=15, captions=CAPTIONS))
