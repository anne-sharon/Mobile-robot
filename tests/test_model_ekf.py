"""Checks on the state-space model, the EKF and the LQR (no rendering needed)."""
import math
import numpy as np
from src import model
from src.estimator import Estimator
from src.control import gain, lqr_gain
from src.world import WHEEL_R, WHEEL_SEP, camera_intrinsics, CAM_POS, ARUCO_DICT, MARKER_ID, MARKER_SIZE

CAL = dict(wheel_radius=WHEEL_R, wheel_separation=WHEEL_SEP, camera_matrix=camera_intrinsics(), camera_pos=CAM_POS,
           marker=dict(dictionary=ARUCO_DICT, id=MARKER_ID, size=MARKER_SIZE))


def _num_jac(f, x, e=1e-6):
    return np.column_stack([(f(x + e * np.eye(len(x))[i]) - f(x)) / e for i in range(len(x))])


def test_jacobians_match_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s, (ds, dth) = rng.normal(size=3), rng.normal(size=2) * 0.1
        F, G = model.odometry_jacobians(s, ds, dth)
        assert np.allclose(F, _num_jac(lambda x: model.odometry_step(x, ds, dth), s), atol=1e-4)
        assert np.allclose(G, _num_jac(lambda u: model.odometry_step(s, u[0], u[1]), np.array([ds, dth])), atol=1e-4)
        assert np.allclose(model.marker_jacobian(s), _num_jac(model.marker_measurement, s), atol=1e-4)


def test_pose_from_marker_inverts_measurement():
    for s in ([1.0, 0.2, math.pi - 0.1], [0.5, -0.3, -3.0], [2.0, 0.0, math.pi]):
        s = np.array(s)
        back = model.pose_from_marker(model.marker_measurement(s))
        assert np.allclose(back[:2], s[:2]) and abs((back[2] - s[2] + math.pi) % (2 * math.pi) - math.pi) < 1e-9


def test_controllability_and_observability():
    a = model.analysis()
    assert a["pose_error_controllability_rank(v_r=0.15)"] == 3
    assert a["pose_error_controllability_rank(v_r=0.0)"] == 2
    assert a["lateral_controllability_rank"] == 2
    assert a["pose_observability_rank_camera"] == 3


def test_lqr_closed_loop_stable():
    for v in (0.02, 0.1, 0.22):
        K, poles = model.lqr(*model.tracking_error_model(v), np.diag([400, 33]), np.array([[1.0]]))
        assert np.all(poles.real < 0)
    assert np.allclose(gain(0.12), lqr_gain(0.12), atol=0.05)


def test_ekf_converges_to_truth_from_measurements():
    """Robot parked 1 m in front of the marker; noisy measurements pull a wrong initial estimate to the truth."""
    est = Estimator(CAL)
    truth = np.array([1.0, 0.05, math.pi - 0.03])
    z_true = model.marker_measurement(truth)
    rng = np.random.default_rng(1)
    est.correct(z_true + rng.normal(0, [0.02, 0.02, 0.03]), t=0.0)      # first fix: dock frame
    assert est.frame == "dock"
    for k in range(60):
        est.predict((0.0, 0.0), 0.0, 0.05 * (k + 1))                     # standing still
        est.correct(z_true + rng.normal(0, [0.004, 0.003, 0.008]), t=0.05 * (k + 1))
    assert np.linalg.norm(est.pose[:2] - truth[:2]) < 0.01
    assert np.all(np.linalg.eigvalsh(est.P) > 0)


def test_ekf_rejects_flipped_measurement():
    est = Estimator(CAL)
    truth = np.array([1.0, 0.0, math.pi])
    z = model.marker_measurement(truth)
    est.correct(z, 0.0)
    for k in range(10):
        est.predict((0.0, 0.0), 0.0, 0.05 * (k + 1)); est.correct(z, 0.05 * (k + 1))
    before = est.pose.copy()
    flipped = z + np.array([0.0, 0.0, math.radians(40)])
    assert est.correct(flipped, 1.0) is False
    assert np.allclose(est.pose, before)
