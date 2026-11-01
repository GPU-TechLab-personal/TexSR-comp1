"""Build-time check for texsr/torchbase (run with `python -I`, from the Dockerfile).

  1. every package named in constraints.txt that is installed is at that version
  2. torch / torchvision are still the base image's 2.0.0 / 0.15.0
  3. scipy's interpolate.interp2d works when called and scipy.finfo exists
     (KAIR utils/utils_blindsr.py:12, :195)
  4. no installed package owns basicsr / models / utils / data, the names the
     official SR submodules provide (variants/util/paths.py)
"""
import importlib.metadata as md
import importlib.util
import re
import sys


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def main(constraints):
    bad = []
    pinned = {}
    for line in open(constraints):
        line = line.split(" #")[0].strip()
        if line and not line.startswith("#"):
            name, ver = line.split("==")
            pinned[norm(name)] = ver
    installed = {norm(d.metadata["Name"]): d.version for d in md.distributions()}
    checked = 0
    for name, ver in sorted(pinned.items()):
        if name in installed:
            checked += 1
            if installed[name] != ver:
                bad.append("%s %s, pinned %s" % (name, installed[name], ver))

    import torch
    import torchvision
    if not torch.__version__.startswith("2.0.0"):
        bad.append("torch " + torch.__version__)
    if not torchvision.__version__.startswith("0.15.0"):
        bad.append("torchvision " + torchvision.__version__)
    # Called, not just imported: from 1.14 on the name still imports and raises
    # NotImplementedError when used (measured in texsr/matup, scipy 1.15.3).
    try:
        import numpy as np
        from scipy.interpolate import interp2d
        interp2d(np.arange(3.0), np.arange(3.0), np.zeros((3, 3)))(0.5, 0.5)
    except Exception as e:  # noqa: BLE001
        bad.append("scipy.interpolate.interp2d unusable: %s: %s" % (type(e).__name__, str(e)[:80]))
    import scipy
    if not hasattr(scipy, "finfo"):   # KAIR utils_blindsr.py:195, gone from scipy 1.12
        bad.append("scipy.finfo missing (scipy %s)" % scipy.__version__)
    taken = [n for n in ("basicsr", "models", "utils", "data") if importlib.util.find_spec(n)]
    if taken:
        bad.append("installed packages shadow the submodules: %s" % taken)

    if bad:
        print("torchbase check FAILED:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("torchbase OK | %d pinned packages installed and matching | torch %s | torchvision %s"
          % (checked, torch.__version__, torchvision.__version__))


if __name__ == "__main__":
    main(sys.argv[1])
