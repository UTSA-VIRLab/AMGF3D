#!/usr/bin/env python
# ============================================================================
# Cross-method robustness stats: ImVoteNet vs TR3D+FF vs AMGF3D on SUN RGB-D.
#
# Reads crossmethod_robustness_sun.csv (produced by crossmethod_robustness_eval.sh)
# and reports % AP RETAINED under each corruption =
#     100 * perturbed_mAP / clean_mAP
# where clean = the points_xyz_noise @ 0.0 (identity) run for each method.
#
# Why % retained (not absolute AP): the methods have different clean accuracy
# (ImVoteNet ~63 << AMGF3D ~70), so absolute robust AP is an unfair comparison;
# % retained measures which method DEGRADES LEAST -- the robustness claim.
# Higher % retained = more robust. The headline rows use the same representative
# levels as main-paper Table 4.
# ============================================================================
import csv, os, sys

BASE = os.environ.get("ROBUSTNESS_DIR",
                      sys.argv[1] if len(sys.argv) > 1 else "results/robustness_eval")
SRC  = os.path.join(BASE, "crossmethod_robustness_sun.csv")
METHODS = ["imvotenet", "tr3d_ff", "amgf3d"]
PRETTY  = {"imvotenet": "ImVoteNet", "tr3d_ff": "TR3D+FF", "amgf3d": "AMGF3D"}
# representative (perturbation, level) matching Table 4; levels as floats (CSV stores e.g. "3.0")
HEADLINE = [("points_subsample", 0.5), ("rgb_gaussian", 0.1),
            ("rgb_dropout", 0.5), ("calibration_rotation", 3.0),
            ("points_xyz_noise", 0.1)]
CLEAN_KEY = ("points_xyz_noise", 0.0)

def load():
    if not os.path.exists(SRC):
        print(f"[error] missing {SRC} -- run crossmethod_robustness_eval.sh first"); return None
    D = {}  # (method, pert, level) -> (m25, m50)
    for r in csv.DictReader(open(SRC)):
        if r.get("dataset") != "sunrgbd":
            continue
        D[(r["method"], r["perturbation"], float(r["level"]))] = (
            float(r["mAP_0.25"]), float(r["mAP_0.50"]))
    return D

def fmt_level(pert, lvl):
    return {"points_subsample": f"subsample {lvl}", "rgb_gaussian": f"RGB-Gaussian s={lvl}",
            "rgb_dropout": f"RGB-dropout {lvl}", "calibration_rotation": f"calib {lvl}deg",
            "points_xyz_noise": f"pt-noise s={lvl}"}.get(pert, f"{pert} {lvl}")

def main():
    D = load()
    if D is None: return
    # clean baselines
    clean = {}
    for m in METHODS:
        if (m,) + CLEAN_KEY in D:
            clean[m] = D[(m,) + CLEAN_KEY]
        else:
            clean[m] = None
            print(f"[warn] no clean baseline for {m} (run points_xyz_noise@0.0)")
    print("Clean mAP@0.25 / @0.50:")
    for m in METHODS:
        c = clean[m]
        print(f"  {PRETTY[m]:10s}: {('%.1f / %.1f'%(c[0]*100,c[1]*100)) if c else 'MISSING'}")

    for thr, idx in [("@0.25", 0), ("@0.50", 1)]:
        print(f"\n=== % AP retained under perturbation ({thr}) -- higher = more robust ===")
        hdr = f"{'corruption':20s} " + " ".join(f"{PRETTY[m]:>10s}" for m in METHODS) + "   most-robust"
        print(hdr); print("-"*len(hdr))
        for pert, lvl in HEADLINE:
            cells, vals = [], {}
            for m in METHODS:
                k = (m, pert, lvl); c = clean[m]
                if k in D and c and c[idx] > 0:
                    ret = 100.0 * D[k][idx] / c[idx]; vals[m] = ret; cells.append(f"{ret:9.1f}%")
                else:
                    cells.append(f"{'--':>10s}")
            best = max(vals, key=vals.get) if vals else None
            print(f"{fmt_level(pert,lvl):20s} " + " ".join(cells) + f"   {PRETTY.get(best,'-')}")
    print("\nClaim to make IF AMGF3D wins most rows: 'Among representative RGB-D fusion")
    print("detectors, AMGF3D retains the most accuracy under corruption (highest % AP")
    print("retained), i.e., degrades least.' Report this as a cross-architecture comparison.")

if __name__ == "__main__":
    main()
