<#
.SYNOPSIS
    The fixed arm: PSNR pretrain and ESRGAN train with the code defects corrected.

.DESCRIPTION
    Stages moved out of scripts/deeptex/run_deeptex.ps1 with their names and
    parameters unchanged (variants/deeptex/fixed/README.md). The trainer is
    paper_loop/main.py --experiment fixed, which takes the old
    deeptex_train.py flags unchanged (L1).

.EXAMPLE
    variants\deeptex\fixed\run.ps1 -Stage pretrain
    variants\deeptex\fixed\run.ps1 -Stage train -Config variants/deeptex/fixed/esrgan_scaled.yaml
#>
[CmdletBinding()]
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("preflight","pretrain","train")]
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
$expCompose = Join-Path $repo "docker\exp\compose.yaml"
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

. (Join-Path $repo "variants\util\lib.ps1")

# Moved from (old line ranges); paths per variants/tests/l5_deeptex.ps1:
#   Stage-Train              run_deeptex.ps1:351-379

function Stage-Train {
    param([string]$which, [string]$CfgOverride = "", [string[]]$Extra = @(), [string]$Label = "")
    $cfg      = if ($CfgOverride) { $CfgOverride }
                elseif ($which -eq "psnr") { $PsnrConfig } else { $Config }
    if (-not $Label) { $Label = $which }
    $subName  = (Select-String -Path (Join-Path $repo $cfg) -Pattern "^sub_name:\s*'(.+)'").Matches.Groups[1].Value
    $niter    = if ($MaxSteps -gt 0) { $MaxSteps } else {
                  [int](Select-String -Path (Join-Path $repo $cfg) -Pattern "^niter:\s*(\d+)").Matches.Groups[1].Value }
    if ($Force) { Clear-Checkpoints $subName }

    $extra = @() + $Extra
    if ($MaxSteps -gt 0) { $extra += @("--max-steps","$MaxSteps") }

    $r = Invoke-Step "$Label training ($niter steps, sub_name=$subName)" deeptex `
        (@("python","/workspace/variants/deeptex/paper_loop/main.py","--experiment","fixed","--cfg_path","/workspace/$cfg",
           "--stage",$which,"--tfrecord",$TFREC,
           "--val-hr","$DS/HR_val","--val-lr","$DS/LRx4_val") + $extra) "train-$Label"
    $kind = Get-FailureKind $r.Text $r.Code
    switch ($kind) {
        "cpu-fallback" { throw "TF fell back to CPU - the run would take weeks. Check the GPU reservation. (see $($r.Log))" }
        "gpu-lost"     { throw "GPU fell off the bus. Reboot before retrying (docs/OPEN-ITEMS.md #10). (see $($r.Log))" }
        "oom"          { throw "out of memory. Lower batch_size in $cfg - the persistent tape at train_esrgan.py:86 is why the footprint is ~3x a plain step. (see $($r.Log))" }
        "nan"          { throw "non-finite loss; a debug checkpoint was saved. Try --losses corrected, or check the pretrain stage. (see $($r.Log))" }
        "data"         { throw "TFRecord problem - re-run texsr_verify_tfrecord.py. (see $($r.Log))" }
        "ok"           { Assert-TrainingReached $r.Text $niter "$Label training" }
        default        { throw "$Label training: $kind (see $($r.Log))" }
    }
    Say "  $Label training OK" "Green"
}

# ---------------------------------------------------------------- main

Say "TexSR / DeepTextures   stage=$Stage dataset=$Dataset   logs -> runs\_logs\$stamp" "White"
if (-not $SkipPreflight) { Invoke-DeeptexPreflight } else { Say "preflight SKIPPED" "Yellow" }

switch ($Stage) {
    "preflight" { }
    "pretrain"  { Stage-Train "psnr" }
    "train"     { Stage-Train "esrgan" }
}

Say "`ndone. logs in runs\_logs\$stamp" "Green"
