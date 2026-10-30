"""Fill the numbers on the project page from the result files (so every number on the page comes from the code).
Run from the repo root after eval.run_batch and eval.tune_ekf:  python -m eval.build_page
Reads eval/page_template.html, writes index.html."""
import csv, os
import numpy as np
from eval.run_batch import CONDITIONS
from src.estimator import Q_SCALE, R_SCALE

LABELS = dict(nominal="Nominal", gyro_noise="Gyro noise + bias", camera_noise="Camera pixel noise",
              push="Push near the dock", occlusion="Camera blocked 6 s", combined="Gyro + camera noise + push")
NUM = ["parked", "pos_m", "heading_deg", "time", "collisions", "mean_pose_err_near_dock_m", "final_pose_err_m",
       "mean_nees", "rejected", "reinits"]


def load(cond, d="eval/results"):
    rows = list(csv.DictReader(open(os.path.join(d, f"batch_{cond}.csv"))))
    for r in rows:
        for k in NUM:
            v = r.get(k, "")
            r[k] = (v == "True") if k == "parked" else (float(v) if v not in ("", "None") else float("nan"))
        r["disturbed"] = r.get("disturbed", "") == "True"
    return rows


def stats(rows):
    g = lambda k: np.array([r[k] for r in rows if np.isfinite(r[k])])
    return dict(n=len(rows), parked=int(sum(r["parked"] for r in rows)),
                coll=int(sum(r["collisions"] > 0 for r in rows)),
                disturbed=int(sum(r["disturbed"] for r in rows)),
                pos=100 * g("pos_m").mean(), pos_w=100 * g("pos_m").max(),
                head=g("heading_deg").mean(), head_w=g("heading_deg").max(),
                near=100 * g("mean_pose_err_near_dock_m").mean(), fin=1000 * g("final_pose_err_m").mean(),
                nees=g("mean_nees").mean(), time=g("time").mean(),
                rej=g("rejected").mean(), reinit_runs=int(sum(r["reinits"] > 0 for r in rows)))


def table(all_stats):
    h = ("<table><tr><th>Condition</th><th>Parked</th><th>Parking error, mean / worst</th>"
         "<th>Estimate error near dock</th><th>NEES</th></tr>")
    for c in CONDITIONS:
        s = all_stats[c]
        h += (f"<tr><td>{LABELS[c]}</td><td>{s['parked']}/{s['n']}</td>"
              f"<td>{s['pos']:.1f} / {s['pos_w']:.1f} cm</td><td>{s['near']:.1f} cm</td><td>{s['nees']:.1f}</td></tr>")
    return h + "</table>"


def build(template="eval/page_template.html", out="index.html", subs=None):
    all_stats = {c: stats(load(c)) for c in CONDITIONS}
    tune = list(csv.DictReader(open("eval/results/ekf_tuning.csv")))
    chosen = [r for r in tune if float(r["q_scale"]) == Q_SCALE and float(r["r_scale"]) == R_SCALE][0]
    tot_p = sum(s["parked"] for s in all_stats.values()); tot_n = sum(s["n"] for s in all_stats.values())
    tot_c = sum(s["coll"] for s in all_stats.values())
    nom = all_stats["nominal"]
    html = open(template).read()
    rep = {"__NOM_PARKED__": f"{nom['parked']}/{nom['n']}", "__ALL_PARKED__": f"{tot_p}/{tot_n}",
           "__NOM_POS__": f"{nom['pos']:.1f}", "__NOM_EST__": f"{nom['fin']:.0f}",
           "__BATCH_TABLE__": table(all_stats), "__TOTAL_COLL__": str(tot_c),
           "__WORST_POS__": f"{max(s['pos_w'] for s in all_stats.values()):.1f}",
           "__TUNE_NEES__": f"{float(chosen['mean_nees']):.1f}", "__TUNE_NEAR__": f"{float(chosen['near_dock_err_cm']):.1f}"}
    for k, v in all_stats.items():
        for kk, vv in v.items():
            rep[f"__{k.upper()}_{kk.upper()}__"] = f"{vv:.1f}" if isinstance(vv, float) else str(vv)
    rep.update(subs or {})
    for k, v in rep.items():
        html = html.replace(k, v)
    left = sorted({t for t in __import__("re").findall(r"__[A-Z_]+__", html)})
    open(out, "w").write(html)
    return all_stats, left


if __name__ == "__main__":
    st, left = build()
    for c, s in st.items():
        print(c, {k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()})
    print("placeholders left:", left)
