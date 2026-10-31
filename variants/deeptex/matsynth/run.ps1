<#
.SYNOPSIS
    The MatSynth arms: pairs + one TFRecord per recipe, and the all-maps evaluation.

.DESCRIPTION
    Stages moved out of scripts/deeptex/run_deeptex.ps1 with their names and
    parameters unchanged (variants/deeptex/matsynth/README.md). Training is
    paper\run.ps1 -Stage paper with -PaperConfig/-PaperRoot.

.EXAMPLE
    variants\deeptex\matsynth\run.ps1 -Stage prep-matsynth -Recipe chratio
    variants\deeptex\matsynth\run.ps1 -Stage eval-arms -PaperConfig variants/deeptex/matsynth/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all
#>
[CmdletBinding()]
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("preflight","prep-matsynth","eval-arms")]
    [string]   $Stage,
    [string]   $Dataset       = "textures_db",
    [string]   $SourceDir     = "external/DeepTextures/data/textures_db",
    [string]   $Config        = "variants/deeptex/paper/esrgan_paper.yaml",
    [string]   $PsnrConfig    = "variants/deeptex/fixed/psnr_texsr.yaml",
    [string]   $EvalTag       = "",
    [string]   $CkptDir       = "",
    [string]   $PaperConfig   = "variants/deeptex/paper/esrgan_paper.yaml",
    # Working directory of the upstream trainer, relative to the repo. Its
    # ./checkpoints/<sub_name> lands here, so each arm needs its own.
    [string]   $PaperRoot     = "runs/deeptex/paper",
    # MatSynth training mix for -Stage prep-matsynth. "all" is the paper's 1:1 per
    # map; the others are ablations (DEEPTEX-DEVIATIONS 0.10). See $MatsynthRecipes.
    [ValidateSet("all","chratio","basecolor")]
    [string]   $Recipe        = "all",
    # -Stage paper: copy every checkpoint out as it is saved (upstream keeps 3).
    [switch]   $KeepAllCheckpoints,
    # -Stage eval-arms: already-evaluated arms (runs/deeptex/eval/<tag>) to add as
    # report columns. Comma-separated is fine: powershell -File passes one string.
    [string[]] $ExtraArms     = @(),
    [int]      $MaxSteps      = 0,
    [int]      $CropSize      = 480,
    [int]      $Step          = 240,
    [string]   $LrKernel      = "bicubic",
    [switch]   $Force,
    [switch]   $DeleteCheckpoints,
    [switch]   $SkipPreflight,
    [string[]] $AllowContainers = @(),
    # Render-loss study (runs/montages/deeptex_render_loss/). Appended AFTER the existing
    # parameters so their positional binding does not move.
    # -Stage render-train: variants/deeptex/render_loss/<Arm>.yaml.
    # r0a / r0b are w_render = 0 at seeds 1 / 2; r1 is seed 1 with the calibrated w.
    [ValidateSet("r0a","r0b","r1")]
    [string]   $Arm           = "r0a",
    # -Stage gate-render. No "all" on purpose: calibrate only prints w_render, and
    # a person writes it into the r1 config before render-train can use it.
    [ValidateSet("parity","calibrate","overfit","smoke","premise")]
    [string]   $Gate          = "parity",
    # -Stage eval-render / eval-perceptual: arm tags (runs/deeptex/<tag>) and checkpoint steps.
    # Commas or spaces: powershell -File hands "a,b" over as ONE string, while a
    # call from a PowerShell prompt flattens an array into "a b".
    [string]   $Arms          = "render_r0a,render_r0b,render_r1",
    [string]   $Steps         = "30000,35000,40000",
    # -Stage eval-perceptual only: where svbrdf_perceptual_eval.py renders and runs
    # LPIPS. auto = cuda:0 when the container sees a GPU; cpu creates no CUDA
    # context, so a training run on the card is left its memory.
    [ValidateSet("auto","cpu","cuda")]
    [string]   $EvalDevice    = "auto",
    # -Stage eval-perceptual with -EvalDevice cpu: torch / OpenCV threads in the exp
    # container (svbrdf_perceptual_eval.py --threads). Uncapped, torch takes one per
    # core the Docker VM exposes, beside the training run's input pipeline (the exp
    # service sets no cpus: limit). 4 is a cap, not a measured optimum; 0 = no cap.
    [ValidateRange(0, 256)]
    [int]      $EvalThreads   = 4
)

$ErrorActionPreference = "Stop"
# variants\deeptex\<arm>\run.ps1 -> three levels up
$repo    = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$compose = Join-Path $repo "docker\compose.yaml"
# The experiments image (service exp, working_dir /workspace): the only one with
# both lpips and flip-evaluator. Used by -Stage eval-perceptual alone.
$expCompose = Join-Path $repo "experiments\docker\compose.yaml"
$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir  = Join-Path $repo "runs\_logs\$stamp"
$DS      = "/workspace/runs/deeptex/datasets/$Dataset"
$TFREC   = "$DS/${Dataset}_lrx4.tfrecord"
# Render-loss study. ABSOLUTE container paths only: the deeptex and matup services
# start inside their SUBMODULE (docker/compose.yaml working_dir), so a relative
# output path would land in external/ and dirty it. Not $GATE: PowerShell names are
# case-insensitive, and that would be the -Gate parameter.
$GROUPED   = "/workspace/runs/deeptex/datasets/matsynth_grouped/matsynth_grouped_lrx4.tfrecord"
$MS_PAIRS  = "/workspace/runs/deeptex/datasets/matsynth"
$RGATE     = "/workspace/runs/deeptex/render_gate"
$RSMOKE    = "/workspace/runs/deeptex/_smoke_render"
$EVAL_ROOT = "/workspace/runs/matsynth"
$R1_CFG    = "variants/deeptex/render_loss/r1.yaml"
# The material deeptex_drift.py draws its per-checkpoint sheet for - the one
# runs/montages/deeptex_checkpoints/ already uses, so the sheets line up.
$DRIFT_SHEET = "0844_syntetic_foam"
# Read before the default below fills it in: eval-arms names its output after
# -PaperRoot unless a tag was given explicitly.
$EvalTagGiven = $PSBoundParameters.ContainsKey("EvalTag")
if (-not $EvalTag) { $EvalTag = $Dataset }

. (Join-Path $repo "variants\common\lib.ps1")

# Moved from (old line ranges); paths per variants/tests/l5_deeptex.ps1:
#   Stage-PrepMatsynth       run_deeptex.ps1:545-584
#   Stage-EvalArms           run_deeptex.ps1:586-628

# Pairs from MatSynth's train split -> one TFRecord over the recipe's maps -> verify.
# The degradation check runs first and is a hard gate: it rebuilds the 12
# evaluation materials and requires pixel identity with runs/matsynth, i.e. that
# the network will be trained on exactly the degradation it is scored on.
function Stage-PrepMatsynth {
    $ms  = "/workspace/runs/deeptex/datasets/matsynth"
    $rec = "$ms/matsynth_${Recipe}_lrx4.tfrecord"
    $forceArg = if ($Force) { @("--force") } else { @() }

    $r = Invoke-Step "degradation identity vs the 12 eval materials" dataprep `
        @("python","variants/deeptex/matsynth/matsynth_make_pairs.py","--check-eval") "prepms-check"
    if ($r.Code -ne 0) { throw "degradation check FAILED - do not build the training set (see $($r.Log))" }

    $r = Invoke-Step "pairs from the MatSynth train shards" dataprep `
        (@("python","variants/deeptex/matsynth/matsynth_make_pairs.py","--out",$ms,"--jobs","4") + $forceArg) "prepms-pairs"
    if ($r.Code -ne 0) { throw "pair generation failed (see $($r.Log)); rerunning resumes" }

    # ONE record, one seeded shuffle across all its maps: upstream reads with
    # shuffle=False, so the record order is the curriculum. The recipe table
    # (old $script:MatsynthRecipes) and its --repeat expansion live in recipes.py,
    # which prints the same "recipe <name> : ..." line; for "all" the record is
    # byte-for-byte the one the old chain wrote (L2).
    $r = Invoke-Step "build one TFRecord ($Recipe)" deeptex `
        @("python","/workspace/variants/deeptex/matsynth/recipes.py","--recipe",$Recipe,"--root",$ms,
          "--out",$rec,"--pairs-manifest","$ms/pairs_manifest.json","--overwrite") "prepms-tfrecord"
    if ($r.Code -ne 0) { throw "TFRecord build failed (see $($r.Log))" }

    # LR here is area-decimated from the 4K source, not imresize_np(HR), so the
    # alignment probe reports a nonzero max as a warning by design; a small
    # sample is enough to catch a gross misalignment.
    $r = Invoke-Step "verify TFRecord" deeptex `
        @("python","/workspace/variants/deeptex/upstream_runner/texsr_verify_tfrecord.py","--tfrecord",$rec,
          "--align-sample","16") "prepms-verify"
    if ($r.Code -ne 0) { throw "VERIFY FAILED - do not train on this dataset (see $($r.Log))" }
}

# Evaluation for a non-default arm. Deliberately NOT Stage-Eval: that one runs
# matup_postprocess + matup_montage --collect, which rewrite compare/<stamp>/ and
# the committed runs/montages/. This writes only under runs/deeptex/eval/ and
# runs/deeptex/report_<tag>/.
function Stage-EvalArms {
    $tag = if ($script:EvalTagGiven) { $EvalTag } else { Split-Path -Leaf $PaperRoot }
    $sn  = (Select-String -Path (Join-Path $repo $PaperConfig) -Pattern "^sub_name:\s*'(.+)'").Matches.Groups[1].Value
    $armOut = "/workspace/runs/deeptex/eval/$tag"
    # Other arms already evaluated by an earlier eval-arms run join the same table.
    # Checked before the SR passes, not after, so a typo costs nothing. Split on
    # commas too: powershell -File hands "a,b" over as ONE string.
    $extra = @($ExtraArms | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
    foreach ($t in $extra) {
        if (-not (Test-Path (Join-Path $repo "runs\deeptex\eval\$t"))) {
            throw "-ExtraArms $t : runs\deeptex\eval\$t does not exist - run -Stage eval-arms for that arm first"
        }
    }
    $r = Invoke-Step "super-resolve all maps + render with $PaperRoot" deeptex `
        @("python","/workspace/variants/deeptex/matsynth/eval_arms.py","--cfg","/workspace/$PaperConfig",
          "--ckpt-dir","/workspace/$PaperRoot/checkpoints/$sn","--root","/workspace/runs/matsynth",
          "--out",$armOut) "evalarms-sr-$tag"
    if ($r.Code -ne 0) { throw "SR failed for $tag (see $($r.Log))" }

    # The baseline: the textures_db checkpoint on the same six inputs. Its
    # existing eval/paper holds baseColor and render only.
    $baseCfg = "variants/deeptex/paper/esrgan_paper.yaml"
    $baseSn  = (Select-String -Path (Join-Path $repo $baseCfg) -Pattern "^sub_name:\s*'(.+)'").Matches.Groups[1].Value
    $baseOut = "/workspace/runs/deeptex/eval/paper_allmaps"
    $r = Invoke-Step "super-resolve all maps + render with the textures_db checkpoint" deeptex `
        @("python","/workspace/variants/deeptex/matsynth/eval_arms.py","--cfg","/workspace/$baseCfg",
          "--ckpt-dir","/workspace/runs/deeptex/paper/checkpoints/$baseSn","--root","/workspace/runs/matsynth",
          "--out",$baseOut) "evalarms-sr-paper_allmaps"
    if ($r.Code -ne 0) { throw "SR failed for the textures_db baseline (see $($r.Log))" }

    $armArgs = @("--arm","textures_db=$baseOut")
    foreach ($t in $extra) { $armArgs += @("--arm","$t=/workspace/runs/deeptex/eval/$t") }
    $armArgs += @("--arm","$tag=$armOut")
    $r = Invoke-Step "arms report" dataprep `
        (@("python","variants/deeptex/matsynth/deeptex_arms_report.py") + $armArgs +
         @("--out","/workspace/runs/deeptex/report_$tag")) "evalarms-report"
    if ($r.Code -ne 0) { throw "report failed (see $($r.Log))" }
    Say "  report -> runs\deeptex\report_$tag\summary.md" "Green"
}

# ---------------------------------------------------------------- main

Say "TexSR / DeepTextures   stage=$Stage dataset=$Dataset   logs -> runs\_logs\$stamp" "White"
if (-not $SkipPreflight) { Invoke-DeeptexPreflight } else { Say "preflight SKIPPED" "Yellow" }

switch ($Stage) {
    "preflight" { }
    "prep-matsynth" { Stage-PrepMatsynth }
    "eval-arms" { Stage-EvalArms }
}

Say "`ndone. logs in runs\_logs\$stamp" "Green"
