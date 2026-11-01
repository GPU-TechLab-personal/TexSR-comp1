#!/usr/bin/env python3
"""Gate parity: write_arm.py's predict() gives what the OFFICIAL inference scripts
give, to 1 LSB, on the evaluation inputs. Runs in the srft service:

    docker compose run --rm srft python variants/sr_finetune/gates/gate_parity.py

-> runs/sr_ft/gate/parity.json ("pass" is what Show-GateResults reads).

Inputs: --materials eval materials x every job (baseColor, normal, roughness,
metallic, height, render_lr), each encoded once with map_encode.to_rgb8 - the
encoding write_arm.py uses - and written as an 8-bit RGB PNG, which is all the
official scripts read (cv2.imread IMREAD_COLOR / 255). That arm_output.load_as_rgb01
gives the same array from the original file is checked too.

Per zero-shot arm, in a child process holding only that code base, the official
script runs unmodified (runpy, cwd = a scratch folder; it writes results/ there):

  esrgan_zs        BasicSR inference/inference_esrgan.py --model_path --input --output
  swinir_cls_zs    SwinIR main_test_swinir.py --task classical_sr --scale 4
                   --training_patch_size 64 (it reads the LR as <name>x4.png beside a
                   GT folder it scores against; the GT here is a blank placeholder)
  swinir_real_zs   main_test_swinir.py --task real_sr --scale 4
  swinir_realL_zs  main_test_swinir.py --task real_sr --scale 4 --large_model

then write_arm's load_model + predictor + arm_output.super_resolve (tile 0) on the
same arrays, quantised as save_colour does (rint(clip(x) * 255)). Pass: every
pixel within 1 of the official PNG.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import runpy
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.util import arm_output, paths  # noqa: E402
from variants.sr_finetune import nets  # noqa: E402
from variants.sr_finetune import write_arm  # noqa: E402
from variants.sr_finetune.gates import probe as child  # noqa: E402

GATE_DIR = os.path.join(paths.REPO, "runs", "sr_ft", "gate")
SCRATCH = os.path.join(GATE_DIR, "parity_scratch")
TOL = 1
ARMS = {
    "esrgan_zs": (os.path.join(paths.BASICSR_ROOT, "inference", "inference_esrgan.py"),
                  lambda w, s: ["--model_path", w, "--input", s + "/lq", "--output", s + "/out"],
                  "out/{}_ESRGAN.png"),
    "swinir_cls_zs": (os.path.join(paths.SWINIR_ROOT, "main_test_swinir.py"),
                      lambda w, s: ["--task", "classical_sr", "--scale", "4", "--training_patch_size", "64",
                                    "--model_path", w, "--folder_lq", s + "/lq_x4", "--folder_gt", s + "/gt"],
                      "results/swinir_classical_sr_x4/{}_SwinIR.png"),
    "swinir_real_zs": (os.path.join(paths.SWINIR_ROOT, "main_test_swinir.py"),
                       lambda w, s: ["--task", "real_sr", "--scale", "4", "--model_path", w,
                                     "--folder_lq", s + "/lq"],
                       "results/swinir_real_sr_x4/{}_SwinIR.png"),
    "swinir_realL_zs": (os.path.join(paths.SWINIR_ROOT, "main_test_swinir.py"),
                        lambda w, s: ["--task", "real_sr", "--scale", "4", "--large_model", "--model_path", w,
                                      "--folder_lq", s + "/lq"],
                        "results/swinir_real_sr_x4_large/{}_SwinIR.png"),
}


def prepare(materials):
    """Encode the inputs once; write them where each official script looks."""
    import cv2
    import numpy as np
    from variants.util import map_encode
    for d in ("lq", "lq_x4", "gt"):
        os.makedirs(os.path.join(SCRATCH, d), exist_ok=True)
    names, same = [], True
    for mat_dir in arm_output.materials(os.path.join(paths.REPO, "runs", "matsynth"))[:materials]:
        mat = os.path.basename(mat_dir)
        run_dir = arm_output.newest_run(mat_dir, mat, "")
        for key, where, fname, _ in arm_output.ALL_JOBS:
            src = os.path.join(mat_dir, "SVBRDF", fname) if where == "svbrdf" else os.path.join(run_dir, fname)
            rgb8 = map_encode.to_rgb8(cv2.imread(src, cv2.IMREAD_UNCHANGED))
            lr01, _ = arm_output.load_as_rgb01(src)
            same = same and np.array_equal(lr01, rgb8.astype(np.float32) / 255.0)
            name = "%s__%s" % (mat, key)
            bgr = rgb8[:, :, ::-1]
            cv2.imwrite(os.path.join(SCRATCH, "lq", name + ".png"), bgr)
            cv2.imwrite(os.path.join(SCRATCH, "lq_x4", name + "x4.png"), bgr)
            h, w = rgb8.shape[:2]
            cv2.imwrite(os.path.join(SCRATCH, "gt", name + ".png"), np.zeros((h * 4, w * 4, 3), np.uint8))
            names.append(name)
    return names, same


def probe(arm, names):
    """Child: the official script, then write_arm's path, on the same inputs."""
    import cv2
    import numpy as np
    script, argv, out_pattern = ARMS[arm]
    arch, wfile, key, _ = write_arm.resolve(arm, "", None)
    work = os.path.join(SCRATCH, arm)
    os.makedirs(work, exist_ok=True)
    for d in ("lq", "lq_x4", "gt"):
        link = os.path.join(work, d)
        if not os.path.exists(link):
            shutil.copytree(os.path.join(SCRATCH, d), link)

    write_arm.USE[nets.ARCHS[arch]["code"]]()
    cwd, sys.argv = os.getcwd(), [script] + argv(wfile, work)
    os.chdir(work)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            runpy.run_path(script, run_name="__main__")
    finally:
        os.chdir(cwd)

    net, device = write_arm.load_model(arch, wfile, key)
    predict = write_arm.predictor(net, nets.ARCHS[arch]["kwargs"].get("window_size"), device)
    per, worst = {}, 0
    for name in names:
        off = cv2.imread(os.path.join(work, out_pattern.format(name)), cv2.IMREAD_COLOR)
        if off is None:
            per[name] = {"error": "official output missing"}
            worst = 255
            continue
        lr = cv2.imread(os.path.join(work, "lq", name + ".png"), cv2.IMREAD_COLOR)[:, :, ::-1]
        sr = arm_output.super_resolve(predict, lr.astype(np.float32) / 255.0, 4, 0, 16)
        ours = np.rint(np.clip(sr, 0.0, 1.0) * 255.0).astype(np.uint8)
        d = np.abs(ours.astype(np.int16) - off[:, :, ::-1].astype(np.int16))
        per[name] = {"max": int(d.max()), "pixels_over_0": int((d > 0).sum()),
                     "pixels_over_tol": int((d > TOL).sum())}
        worst = max(worst, int(d.max()))
    child.emit({"ok": worst <= TOL, "max_abs_diff": worst, "images": per,
                "official": os.path.relpath(script, paths.REPO), "official_log_tail": log.getvalue()[-600:]})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--materials", type=int, default=2, help="how many eval materials (x6 inputs each)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--probe", choices=sorted(ARMS), help=argparse.SUPPRESS)
    ap.add_argument("--names", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe:
        return probe(args.probe, args.names.split(","))

    shutil.rmtree(SCRATCH, ignore_errors=True)
    names, same = prepare(args.materials)
    checks = {"encoding": {"ok": same, "inputs": len(names),
                           "note": "load_as_rgb01(original) == to_rgb8(original) / 255 on every input"}}
    for arm in args.arms.split(","):
        checks[arm] = child.run(os.path.abspath(__file__), arm, "--names", ",".join(names))
        print("  %-16s max |diff| %s" % (arm, checks[arm].get("max_abs_diff", "n/a")))
    shutil.rmtree(SCRATCH, ignore_errors=True)

    report = {"tolerance": TOL, "checks": checks, "failed": [n for n, c in checks.items() if not c["ok"]]}
    report["pass"] = not report["failed"]
    path = os.path.join(GATE_DIR, "parity.json")
    with open(path + ".partial", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(path + ".partial", path)
    print("[gate parity] %s -> %s" % ("PASS" if report["pass"] else "FAIL", path))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
