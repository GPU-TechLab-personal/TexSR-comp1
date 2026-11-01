"""Colour drift of an SR model, step by step - model-agnostic (numpy + cv2).

The offset definition: per image, mean(SR) - mean(HR) per RGB channel in 8-bit
levels, SR clipped to [0,1] and taken as float (not rounded to a PNG); summarised
as R, G, B, brightness (their mean), cast G - (R+B)/2 and the mean over images
of the largest channel offset. LR and HR have the same channel means by
construction (area decimation), so any offset is something the network added.

A method's driver restores each of its checkpoints, wraps the model as
predict(NHWC float32) -> NHWC, and calls offsets() / sheet_outputs(); the tables
and the sheet come from here, so every method's drift reads the same.
variants/deeptex/matsynth/deeptex_drift.py is the DeepTextures (TF) driver.
"""
import json
import os

import cv2
import numpy as np

from variants.util import arm_output, map_encode
from variants.util.report import images, sheets

SHEET_MAPS = [j[0] for j in arm_output.ALL_JOBS]   # baseColor ... height, render


def rgb8(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise SystemExit("cannot read " + path)
    return map_encode.to_rgb8(a)


def offsets(predict, items, scale=4):
    """items: [(lr_path, hr_path)] -> per-image mean(SR) - mean(HR), RGB, 8-bit levels."""
    out = []
    for lp, hp in items:
        lr = rgb8(lp).astype(np.float32) / 255.0
        sr = np.clip(arm_output.super_resolve(predict, lr, scale, 0, 16), 0.0, 1.0) * 255.0
        hr = rgb8(hp).astype(np.float64)
        out.append(sr.reshape(-1, 3).mean(0) - hr.reshape(-1, 3).mean(0))
    return np.array(out)


def summarise(b):
    m = b.mean(0)
    return {"n": int(len(b)), "R": float(m[0]), "G": float(m[1]), "B": float(m[2]),
            "brightness": float(m.mean()), "cast": float(m[1] - (m[0] + m[2]) / 2.0),
            "max_abs_offset": float(np.abs(b).max(1).mean())}


def sheet_outputs(predict, mat_dir, out_dir, scale=4):
    """One material's six outputs through the arm writer's own encoding and
    collapse (arm_output.save_sr, 'mean'), written to out_dir and read back."""
    name = os.path.basename(mat_dir.rstrip("/"))
    run_dir = arm_output.newest_run(mat_dir, name, "")
    os.makedirs(out_dir, exist_ok=True)
    got = {}
    for key, where, fname, kind in arm_output.ALL_JOBS:
        src = os.path.join(mat_dir, "SVBRDF", fname) if where == "svbrdf" else \
            (os.path.join(run_dir, fname) if run_dir else "")
        if not src or not os.path.exists(src):
            continue
        lr, _ = arm_output.load_as_rgb01(src)
        sr = arm_output.super_resolve(predict, lr, scale, 0, 16)
        p = os.path.join(out_dir, key + ".png")
        arm_output.save_sr(p, sr, kind, "mean")
        got[key] = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    return got


def write_sheet(path, title, label, outputs):
    """One row: the checkpoint's six outputs, label lines on the left."""
    cell, pad, left, top, head = 320, 6, 150, 32, 34
    w = left + len(SHEET_MAPS) * (cell + pad) + pad
    s = np.full((top + head + cell + pad, w, 3), 18, np.uint8)
    cv2.putText(s, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for c, m in enumerate(SHEET_MAPS):
        x = left + c * (cell + pad)
        cv2.putText(s, m, (x + 4, top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (210, 210, 210), 1, cv2.LINE_AA)
        a = outputs.get(m)
        s[top + head:top + head + cell, x:x + cell] = (
            images.to_display(a, cell) if a is not None else sheets.placeholder(cell, ["missing"]))
    for i, t in enumerate(label):
        cv2.putText(s, t, (10, top + head + cell // 2 - 8 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    sheets.fit(t, left - 18, 0.62), (210, 210, 210), 1, cv2.LINE_AA)
    if not cv2.imwrite(path, s):
        raise RuntimeError("cv2.imwrite failed for " + path)


def print_row(row):
    t, e = row["train"], row["eval"]
    print("  %-8s step %6d | train R %+6.2f G %+6.2f B %+6.2f cast %+5.2f | eval R %+6.2f G %+6.2f B %+6.2f cast %+5.2f"
          % (row["checkpoint"], row["step"], t["R"], t["G"], t["B"], t["cast"],
             e["R"], e["G"], e["B"], e["cast"]))


def write_report(out, header, rows, n_train, n_eval, pairs, seed, eval_map, source):
    """drift.json (header keys first, then eval_materials and checkpoints) and drift.md.
    `source` names where the checkpoints came from (the ckpt dir)."""
    rep = dict(header)
    rep.update({"eval_materials": n_eval, "checkpoints": rows})
    with open(os.path.join(out, "drift.json"), "w") as f:
        json.dump(rep, f, indent=1)
    lines = ["# Colour drift per checkpoint", "",
             "mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean",
             "over images of the largest channel offset. train = %d seeded pairs from `%s` (seed %d);"
             % (n_train, pairs, seed),
             "eval = %d materials' `%s`. Checkpoints from `%s`." % (n_eval, eval_map, source), "",
             "| checkpoint | step | train R | G | B | brightness | cast | max\\|off\\| "
             "| eval R | G | B | brightness | cast | max\\|off\\| |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, e = r["train"], r["eval"]
        lines.append("| %s | %d | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %.2f | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %.2f |"
                     % (r["checkpoint"], r["step"], t["R"], t["G"], t["B"], t["brightness"], t["cast"],
                        t["max_abs_offset"], e["R"], e["G"], e["B"], e["brightness"], e["cast"],
                        e["max_abs_offset"]))
    with open(os.path.join(out, "drift.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("%d checkpoint(s) -> %s" % (len(rows), os.path.join(out, "drift.md")))
