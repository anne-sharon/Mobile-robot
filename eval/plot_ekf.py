"""Plots for the project page (evaluation only).
Run from the repo root:  python -m eval.plot_ekf
  eval/results/ekf_error.png   : one nominal run -- EKF error in x, y, heading with the +-2 sigma band, and NEES
  eval/results/ekf_tuning.png  : the Q/R tuning trade-off (needs eval/results/ekf_tuning.csv from eval.tune_ekf)
"""
import os, csv, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from eval.tune_ekf import episode
from src.estimator import Q_SCALE, R_SCALE


def plot_error(out="eval/results/ekf_error.png"):
    s, truth = episode((-2.3, -1.2, 2.4), 0, Q_SCALE, R_SCALE)
    rows = truth.rows
    t = np.array([r["t"] for r in rows])
    fr = np.array([r["frame"] for r in rows])
    err = np.array([[r["est_x"] - r["true_x"], r["est_y"] - r["true_y"],
                     (r["est_th"] - r["true_th"] + math.pi) % (2 * math.pi) - math.pi] for r in rows])
    sd = np.array([[r["sd_x"], r["sd_y"], r["sd_th"]] for r in rows])
    nees = np.array([r.get("nees", np.nan) for r in rows])
    t_fix = t[np.argmax(fr == "dock")]
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    names, scale = ["x error [cm]", "y error [cm]", "heading error [deg]"], [100, 100, 180 / math.pi]
    for k in range(3):
        ax[k].fill_between(t, -2 * sd[:, k] * scale[k], 2 * sd[:, k] * scale[k], color="#c9d8f2", label="EKF ±2σ")
        ax[k].plot(t, err[:, k] * scale[k], color="#2a52a0", lw=1.4, label="estimate − truth")
        ax[k].axvline(t_fix, color="#888", lw=1, ls="--")
        ax[k].set_ylabel(names[k]); ax[k].grid(alpha=.3)
        lim = np.nanpercentile(np.abs(err[:, k] * scale[k]), 98) * 1.6 + 1e-3
        ax[k].set_ylim(-lim, lim)
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].set_title(f"EKF on the default start: frame switches start → dock at t = {t_fix:.1f} s (first marker fix)")
    ax[3].plot(t, nees, color="#a03030", lw=1)
    ax[3].axhline(3, color="k", ls=":", lw=1, label="expected (3 states)")
    ax[3].axhline(7.81, color="#888", ls="--", lw=1, label="95% bound")
    ax[3].set_yscale("log"); ax[3].set_ylabel("NEES"); ax[3].set_xlabel("time [s]"); ax[3].grid(alpha=.3)
    ax[3].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return s


def plot_tuning(src="eval/results/ekf_tuning.csv", out="eval/results/ekf_tuning.png"):
    rows = list(csv.DictReader(open(src)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for q in sorted({float(r["q_scale"]) for r in rows}):
        rr = sorted([r for r in rows if float(r["q_scale"]) == q], key=lambda r: float(r["r_scale"]))
        rs = [float(r["r_scale"]) for r in rr]
        ax[0].plot(rs, [float(r["mean_nees"]) for r in rr], "o-", label=f"Q × {q:g}")
        ax[1].plot(rs, [float(r["near_dock_err_cm"]) for r in rr], "o-", label=f"Q × {q:g}")
    ax[0].axhline(3, color="k", ls=":", lw=1); ax[0].set_yscale("log")
    ax[0].set_title("Consistency: mean NEES (3 = consistent)"); ax[1].set_title("Accuracy: pose error within 1.5 m of dock [cm]")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("R scale"); a.grid(alpha=.3); a.legend(fontsize=9)
    ax[0].plot([R_SCALE], [float([r for r in rows if float(r["q_scale"]) == Q_SCALE and float(r["r_scale"]) == R_SCALE][0]["mean_nees"])],
               "k*", ms=14, zorder=5)
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    print(plot_error())
    if os.path.exists("eval/results/ekf_tuning.csv"):
        plot_tuning()
