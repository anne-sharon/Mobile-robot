class Controller:
    """Turn the plan into wheel commands (v [m/s], w [rad/s]).  Empty for now: stands still."""
    def __init__(self, calibration):
        self.cal = calibration

    def update(self, pose, plan):
        return 0.0, 0.0
