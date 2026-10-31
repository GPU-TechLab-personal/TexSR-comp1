#!/usr/bin/env bash
# The whole experiments verification, L0 + L6, into one report.
#
#   variants/tests/l6_experiments_all.sh          # L0 + imports + CPU + GPU   (~15 min)
#   variants/tests/l6_experiments_all.sh cpu      # without the GPU layers     (~4 min)
#
# From Git Bash on the Windows host. Writes
#   $MAIN/runs/variants_verify/experiments/<stamp>/{l0.txt,l0_all.txt,imports.txt,cpu.json,gpu.json,report.md}
# and, for the GPU smoke/compat runs, D:/experimentdata/texsr_exp/variants_verify/expdata/.
set -euo pipefail
cd "$(dirname "$0")/../.."
MAIN=${MAIN:-C:/Users/admin/Desktop/Vsc_Workspace/TexSR-comp1}
stamp=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT="$MAIN/runs/variants_verify/experiments/$stamp"
CT="/workspace/runs/variants_verify/experiments/$stamp"
mkdir -p "$OUT"
rc=0
python variants/tests/l0_static.py --moves variants/tests/moves/experiments.json > "$OUT/l0.txt" || rc=1
python variants/tests/l0_static.py > "$OUT/l0_all.txt" || rc=1
bash variants/tests/run_experiments_l6.sh cpu python variants/tests/l6_experiments_imports.py > "$OUT/imports.txt" || rc=1
bash variants/tests/run_experiments_l6.sh cpu python variants/tests/l6_experiments_cpu.py --out "$CT" || rc=1
if [ "${1:-all}" != cpu ]; then
  bash variants/tests/run_experiments_l6.sh gpu python variants/tests/l6_experiments_gpu.py --out "$CT" || rc=1
fi
python variants/tests/l6_experiments_report.py --out "$OUT" || rc=1
echo "-> $OUT/report.md"
exit $rc
