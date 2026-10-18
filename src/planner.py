from .contracts import Plan

class Planner:
    """What should I do next?  Empty for now."""
    def __init__(self, calibration):
        self.cal = calibration

    def update(self, pose, obs):
        # TODO: search for the marker, avoid obstacles using obs["lidar"], approach and dock
        return Plan(mode="idle")
