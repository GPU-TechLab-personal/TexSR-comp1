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
#   /vvx                           .../variants_verify/expdata           rw  an EXPDATA for runs of the
#     /vvx/materials, /vvx/cache   texsr_exp/materials, texsr_exp/cache  ro  experiment code itself
# The experiment code writes under $EXPDATA (runs/, _gate_train/); the tests that
# run it set EXPDATA=/vvx, so nothing lands in texsr_exp/runs.
set -euo pipefail
mode=$1; shift
HERE=$(cd "$(dirname "$0")/../.." && pwd -W)
MAIN=${MAIN:-C:/Users/admin/Desktop/Vsc_Workspace/TexSR-comp1}
EXPD=${EXPD:-D:/experimentdata}
gpu=()
if [ "$mode" = gpu ]; then
  # another texsr container holding the GPU? (CPU-only ones do not count)
  for id in $(docker ps -q); do
    info=$(docker inspect --format '{{.Config.Image}} {{json .HostConfig.DeviceRequests}}' "$id" 2>/dev/null || true)
    case "$info" in
      *texsr*null*|"") ;;
      *texsr*) echo "a texsr GPU container is running ($info) - wait for it (variants/README.md rule 8)" >&2
               exit 3 ;;
    esac
  done
  gpu=(--gpus all)
fi
VVX="$EXPD/texsr_exp/variants_verify/expdata"
mkdir -p "$MAIN/runs/variants_verify" "$VVX/materials" "$VVX/cache"
MSYS_NO_PATHCONV=1 docker run --rm "${gpu[@]}" --shm-size 8g \
  -v "$HERE:/workspace" \
  -v "$MAIN/external:/workspace/external:ro" \
  -v "$MAIN/runs:/workspace/runs:ro" \
  -v "$MAIN/runs/variants_verify:/workspace/runs/variants_verify" \
  -v "$EXPD:/data:ro" \
  -v "$EXPD/texsr_exp:/expdata:ro" \
  -v "$EXPD/texsr_exp/variants_verify:/vv" \
  -v "$VVX:/vvx" \
  -v "$EXPD/texsr_exp/materials:/vvx/materials:ro" \
  -v "$EXPD/texsr_exp/cache:/vvx/cache:ro" \
  -w /workspace -e PYTHONDONTWRITEBYTECODE=1 \
  texsr/exp:torch2.0.0-cu117 "$@"
