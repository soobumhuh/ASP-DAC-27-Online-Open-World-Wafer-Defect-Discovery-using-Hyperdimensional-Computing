#!/usr/bin/env python3
"""One TSV line per experiment arm: native-grid fold-0 / CV PI-CAI, AUROC, AP, paired-bootstrap delta vs reference."""
import argparse
import json

ap = argparse.ArgumentParser()
for k in ("arm", "native", "boot", "cv", "args"):
    ap.add_argument("--" + k, default="")
a = ap.parse_args()
n = json.load(open(a.native))
b = json.load(open(a.boot))["arm"]
cv = json.load(open(a.cv))
f0 = n["per_fold"][0]["native"]
c = n["native"]
row = [a.arm, f"{f0['picai_score']:.3f}", f"{f0['patient_auroc']:.3f}", f"{f0['lesion_ap']:.3f}",
       f"{c['picai_score'][0]:.3f}±{c['picai_score'][1]:.3f}", f"{c['patient_auroc'][0]:.3f}", f"{c['lesion_ap'][0]:.3f}"]
for part in ("cv_pooled", "fold0"):
    p = b[part]["picai_score"]
    row.append(f"{p['diff_vs_ref']:+.3f} [{p['diff_ci95'][0]:+.3f}, {p['diff_ci95'][1]:+.3f}]")
row.append(f"{cv['cv']['merged']['picai_score'][0]:.3f}")  # crop-grid CV for reference
row.append(json.dumps([r.get("rescorer") for r in cv["per_fold"]]) + " " + a.args)
print("\t".join(row))
