#!/usr/bin/env python3
"""
Gate: each arm config differs from its reference in exactly the factor it
names - nothing else leaks in through inheritance. Seconds, no GPU.

    python variants/experiments/gates/gate_configs.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3)))
from variants.experiments.exp2_esrgan.core.config import load_config   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
C2 = os.path.normpath(os.path.join(_HERE, os.pardir, "exp2_esrgan", "configs")) + os.sep
C3 = os.path.normpath(os.path.join(_HERE, os.pardir, "exp3_render_loss", "configs")) + os.sep

# arm -> (reference arm, the only top-level-path prefixes allowed to differ)
EXPECT = {
    C2 + "2-1_base_pretrained.yaml": (C2 + "2-0_base_scratch.yaml", {"name", "init.mode"}),
    C2 + "2-2_separate.yaml": (C2 + "2-0_base_scratch.yaml", {"name", "model.arch"}),
    C2 + "2-3_mapaware_loss.yaml": (C2 + "2-0_base_scratch.yaml", {"name", "loss.per_map.normal",
                                                                  "loss.per_map.roughness",
                                                                  "loss.per_map.metallic"}),
    C2 + "2-4_concat_input.yaml": (C2 + "2-0_base_scratch.yaml", {"name", "model.input"}),
    C3 + "3-1_render.yaml": (C3 + "3-0_maploss.yaml", {"name", "loss.render", "loss.extra_l1"}),
    C3 + "3-2_render_jointD.yaml": (C3 + "3-1_render.yaml", {"name", "disc.mode", "loss.joint_gan"}),
}


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        key = p + k
        if isinstance(v, dict) and v:
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def main():
    fails = 0
    for arm, (ref, allowed) in EXPECT.items():
        a, r = flat(load_config(arm, [])), flat(load_config(ref, []))
        diff = sorted(k for k in set(a) | set(r) if a.get(k) != r.get(k))
        stray = [k for k in diff if not any(k == p or k.startswith(p + ".") for p in allowed)]
        ok = not stray and diff
        fails += not ok
        print("  %s %-24s vs %-22s differs in: %s%s" % (
            "PASS" if ok else "FAIL", os.path.basename(arm), os.path.basename(ref),
            ", ".join(diff), ("   STRAY: %s" % stray) if stray else ""))
    mapaware = load_config(C2 + "2-3_mapaware_loss.yaml", [])["loss"]["per_map"]
    ok = mapaware["normal"] == {"angular": 1.0} and mapaware["roughness"] == {"l1": 1.0}
    fails += not ok
    print("  %s 2-3 per-map specs replace (not merge): normal=%s roughness=%s" % (
        "PASS" if ok else "FAIL", mapaware["normal"], mapaware["roughness"]))
    print("\n%s" % ("ALL PASS" if not fails else "FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
