#!/usr/bin/env bash
# Run one command of the experiments L6 suite in texsr/exp:torch2.0.0-cu117.
#
#   variants/tests/run_experiments_l6.sh cpu python variants/tests/l6_experiments_cpu.py
#   variants/tests/run_experiments_l6.sh gpu python variants/tests/l6_experiments_gpu.py
#
# From Git Bash on the Windows host. MAIN is the main checkout (runs/, the
# submodule contents and the SwinIR weights live there; a worktree has none).
# Mounts, all read-only except the two variants_verify folders:
#   /workspace                     this checkout (worktree or main)
#   /workspace/external            $MAIN/external                        ro
#   /workspace/runs                $MAIN/runs                            ro
#   /workspace/runs/variants_verify  $MAIN/runs/variants_verify          rw  (reports)
#   /data                          D:/experimentdata                     ro
#   /expdata                       D:/experimentdata/texsr_exp           ro  (materials, caches)
#   /vv                            D:/experimentdata/texsr_exp/variants_verify  rw  (test outputs)
# The experiment code writes under $EXPDATA; tests that run it set EXPDATA=/vv/...
set -euo pipefail
mode=$1; shift
HERE=$(cd "$(dirname "$0")/../.." && pwd -W)
MAIN=${MAIN:-C:/Users/admin/Desktop/Vsc_Workspace/TexSR-comp1}
EXPD=${EXPD:-D:/experimentdata}
gpu=()
if [ "$mode" = gpu ]; then
  if docker ps --format '{{.Image}} {{.Names}}' | grep -i texsr; then
    echo "another texsr container is running - wait for it (variants/README.md rule 8)" >&2
    exit 3
  fi
  gpu=(--gpus all)
fi
mkdir -p "$MAIN/runs/variants_verify" "$EXPD/texsr_exp/variants_verify"
MSYS_NO_PATHCONV=1 docker run --rm "${gpu[@]}" --shm-size 8g \
  -v "$HERE:/workspace" \
  -v "$MAIN/external:/workspace/external:ro" \
  -v "$MAIN/runs:/workspace/runs:ro" \
  -v "$MAIN/runs/variants_verify:/workspace/runs/variants_verify" \
  -v "$EXPD:/data:ro" \
  -v "$EXPD/texsr_exp:/expdata:ro" \
  -v "$EXPD/texsr_exp/variants_verify:/vv" \
  -w /workspace -e PYTHONDONTWRITEBYTECODE=1 \
  texsr/exp:torch2.0.0-cu117 "$@"
