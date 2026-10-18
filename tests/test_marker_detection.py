"""The camera must find the marker and measure its distance from pixels alone."""
import pytest
from src.world import Sim, DOCK_FACE_X, DOCK_Y, CAM_POS
from src.marker import detect_marker

@pytest.mark.parametrize("dist", [0.5, 1.0, 1.5, 2.5])
def test_marker_depth(dist):
    try:
        sim = Sim()
        img_cal = sim.sensors.calibration
        sim.reset(DOCK_FACE_X - dist - CAM_POS[0], DOCK_Y, 0.0)
        img = sim.sensors.camera()
    except Exception as e:                       # no OpenGL available (e.g. plain CI box)
        pytest.skip(f"cannot render: {e}")
    res = detect_marker(img, img_cal["camera_matrix"], img_cal["marker"])
    assert res is not None, "marker not detected"
    right, down, forward = res[2]
    assert abs(forward - dist) < 0.10 * dist + 0.02
    assert abs(right) < 0.02


