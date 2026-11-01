<#
.SYNOPSIS
    The paper arm: textures_db prep -> smoke -> upstream train_esrgan.py -> eval.

.DESCRIPTION
    Stages moved out of scripts/deeptex/run_deeptex.ps1 with their names and
    parameters unchanged (variants/deeptex/paper/README.md). The training
    stage runs external/DeepTextures/train_esrgan.py itself through
    upstream_runner/run_upstream_train.py, exactly as before; the MatSynth arms
    train through this stage too (-PaperConfig/-PaperRoot), and evaluate with
    variants/deeptex/matsynth/run.ps1 -Stage eval-arms.

.EXAMPLE
    variants\deeptex\paper\run.ps1 -Stage prep
    variants\deeptex\paper\run.ps1 -Stage smoke
    variants\deeptex\paper\run.ps1 -Stage paper -KeepAllCheckpoints
    variants\deeptex\paper\run.ps1 -Stage eval -EvalTag paper

.EXAMPLE
    # A MatSynth arm's training (then matsynth\run.ps1 -Stage eval-arms):
    variants\deeptex\paper\run.ps1 -Stage paper -PaperConfig variants/deeptex/matsynth/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all
#>
[CmdletBinding()]
[CmdletBinding()]
param(
    [ValidateSet("preflight","prep","smoke","paper","eval","all")]
    [string]   $Stage         = "all",
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

. (Join-Path $repo "variants\common\lib.ps1")

# Moved from (old line ranges); paths per variants/tests/l5_deeptex.ps1:
#   Stage-Prep               run_deeptex.ps1:291-314
#   Stage-Smoke              run_deeptex.ps1:316-336
#   Stage-Paper              run_deeptex.ps1:381-503
#   Stage-Eval               run_deeptex.ps1:505-529

function Stage-Prep {
    $forceArg = if ($Force) { @("--force") } else { @() }

    $r = Invoke-Step "normalise $SourceDir" dataprep `
        (@("python","variants/deeptex/paper/texsr_normalize.py","--in","/workspace/$SourceDir",
           "--out","$DS/normalized") + $forceArg) "prep-normalize"
    if ($r.Code -ne 0) { throw "normalise failed (see $($r.Log))" }

    $r = Invoke-Step "tile into exact x4 pairs" deeptex `
        (@("python","/workspace/variants/deeptex/paper/texsr_make_pairs.py","--in","$DS/normalized",
           "--out",$DS,"--crop-size","$CropSize","--step","$Step",
           "--lr-kernel",$LrKernel,"--jobs","8") + $forceArg) "prep-pairs"
    if ($r.Code -ne 0) { throw "pair generation failed (see $($r.Log))" }

    $r = Invoke-Step "build TFRecord" deeptex `
        @("python","/workspace/variants/deeptex/upstream_runner/texsr_build_tfrecord.py","--hr","$DS/HR",
          "--lr","$DS/LRx4","--out",$TFREC,"--pairs-manifest","$DS/pairs_manifest.json",
          "--overwrite") "prep-tfrecord"
    if ($r.Code -ne 0) { throw "TFRecord build failed (see $($r.Log))" }

    $r = Invoke-Step "verify TFRecord" deeptex `
        @("python","/workspace/variants/deeptex/upstream_runner/texsr_verify_tfrecord.py","--tfrecord",$TFREC) "prep-verify"
    if ($r.Code -ne 0) { throw "VERIFY FAILED - do not train on this dataset (see $($r.Log))" }
}

function Stage-Smoke {
    # Everything expensive, proven in about four minutes: config paths, CUDA, the
    # VGG19 weights baked into the image, checkpoint writes, and - the part
    # test.py could never do - the train -> eval handoff.
    $r = Invoke-Step "smoke train (20 steps)" deeptex `
        @("python","/workspace/variants/deeptex/paper_loop/main.py","--experiment","fixed","--cfg_path","/workspace/$PsnrConfig",
          "--stage","psnr","--tfrecord",$TFREC,"--sub-name","smoke_psnr",
          "--val-hr","$DS/HR_val","--val-lr","$DS/LRx4_val",
          "--max-steps","20","--save-steps","10","--log-every","10",
          "--val-every","20","--val-count","4") "smoke-train"
    $kind = Get-FailureKind $r.Text $r.Code
    if ($kind -ne "ok") { throw "smoke train: $kind (see $($r.Log))" }
    Assert-TrainingReached $r.Text 20 "smoke train"

    $r = Invoke-Step "smoke eval (self-test + 2 materials)" deeptex `
        @("python","/workspace/variants/deeptex/infer/deeptex_sr.py","--cfg","/workspace/$PsnrConfig",
          "--ckpt-dir","/workspace/runs/deeptex/checkpoints/smoke_psnr","--self-test",
          "--in-dir","/workspace/runs/matsynth/acg_concrete_026/SVBRDF",
          "--out-dir","/workspace/runs/deeptex/eval/_smoke") "smoke-eval"
    if ($r.Code -ne 0) { throw "smoke eval failed (see $($r.Log))" }
}

# The paper arm does NOT use variants/deeptex/paper_loop/main.py. It execs
# external/DeepTextures/train_esrgan.py itself, through a wrapper whose only job is
# to split upstream carriage-return progress bar into newline-terminated [metrics] lines.
# The wrapper reads stdout; it never touches the training process's weights,
# gradients, optimizer or RNG - so the baseline is the released code, not a
# re-implementation claiming equivalence. See docs/DEEPTEX-DEVIATIONS.md 0.4.
function Stage-Paper {
    # $ctrRoot, not $paperRoot: PowerShell names are case-insensitive, so a local
    # $paperRoot IS the -PaperRoot parameter, and assigning it rewrote the
    # parameter to "/workspace/..." before $hostRoot was derived from it.
    $ctrRoot   = "/workspace/$PaperRoot"
    $hostRoot  = Join-Path $repo ($PaperRoot -replace '/', '\')
    New-Item -ItemType Directory -Force -Path $hostRoot | Out-Null

    $cfgPath = "/workspace/$PaperConfig"
    $niter   = [int](Select-String -Path (Join-Path $repo $PaperConfig) -Pattern "^niter:\s*(\d+)").Matches.Groups[1].Value
    $subName = (Select-String -Path (Join-Path $repo $PaperConfig) -Pattern "^sub_name:\s*'(.+)'").Matches.Groups[1].Value

    if ($MaxSteps -gt 0) {
        # Upstream has no --max-steps; the only way to shorten it is niter in the
        # config, so write a scratch copy rather than editing the real one.
        $tmpHost = Join-Path $hostRoot "esrgan_paper_maxsteps.yaml"
        # save_steps must come down too, or a short run writes no checkpoint at all
        # and the verification below rightly fails.
        $half = [Math]::Max(1, [Math]::Floor($MaxSteps / 2))
        (Get-Content (Join-Path $repo $PaperConfig)) `
            -replace "^niter:\s*\d+", "niter: $MaxSteps" `
            -replace "^save_steps:\s*\d+", "save_steps: $half" | Set-Content $tmpHost -Encoding utf8
        $cfgPath = "$ctrRoot/esrgan_paper_maxsteps.yaml"
        $niter   = $MaxSteps
        Say "  -MaxSteps $MaxSteps -> scratch config $cfgPath" "DarkGray"
    }

    $ckpt = Join-Path $hostRoot "checkpoints\$subName"
    if ($Force -and (Test-Path $ckpt)) {
        if ($DeleteCheckpoints) {
            $bak = "$ckpt.bak-$stamp"; Move-Item $ckpt $bak
            Say "  moved existing checkpoints aside -> $bak" "Yellow"
        } else {
            Say "  resuming into existing checkpoints at $ckpt" "DarkGray"
            Say "  (pass -Force -DeleteCheckpoints to start over)" "DarkGray"
        }
    }

    # -KeepAllCheckpoints: upstream keeps only the last three (train_esrgan.py:64,
    # max_to_keep=3), so 5k-25k are gone by the end of a 40k run. A host-side job
    # copies each one out as it is listed, leaving the trainer untouched.
    $keepJob = $null
    if ($KeepAllCheckpoints) {
        $keepSrc = Join-Path $hostRoot "checkpoints\$subName"
        $keepDst = Join-Path $hostRoot "checkpoints_all\$subName"
        $keepJob = Start-CheckpointKeeper $keepSrc $keepDst
    }
    try {
        $r = Invoke-Step "paper arm: upstream train_esrgan.py VERBATIM ($niter steps)" deeptex `
            @("python","/workspace/variants/deeptex/upstream_runner/run_upstream_train.py",
              "--cfg_path",$cfgPath,"--gpu","0","--log-every","100") "train-paper" $ctrRoot
    } finally {
        if ($keepJob) { Stop-CheckpointKeeper $keepJob $keepSrc $keepDst }
    }
    $kind = Get-FailureKind $r.Text $r.Code
    switch ($kind) {
        "cpu-fallback" { throw "TF fell back to CPU - the run would take weeks. (see $($r.Log))" }
        "gpu-lost"     { throw "GPU fell off the bus. Reboot before retrying (docs/OPEN-ITEMS.md #10). (see $($r.Log))" }
        "oom"          { throw "out of memory. Lower batch_size in $PaperConfig. (see $($r.Log))" }
        "data"         { throw "TFRecord problem - re-run texsr_verify_tfrecord.py. (see $($r.Log))" }
        "ok"           { Assert-TrainingReached $r.Text $niter "paper training" }
        default        { throw "paper training: $kind (see $($r.Log))" }
    }
    # The docstring promised a state assertion, not just a log grep. Verify the
    # checkpoint exists and its own step tensor reached niter - a run can print
    # [metrics] and still have written nothing (save_steps above niter, full disk).
    #
    # Build the snippet as ONE variable first. Writing "a" + "b" inline inside an
    # @(...) array literal does NOT concatenate here - each string becomes its own
    # element, python receives only the first as -c, imports tensorflow, prints
    # nothing and exits 0. That is how this very check silently passed while doing
    # nothing, which is the failure mode this whole pipeline exists to prevent.
    $py = "import tensorflow as tf, sys" + "`n" +
          "p = tf.train.latest_checkpoint('checkpoints/$subName')" + "`n" +
          "if not p:" + "`n" +
          "    print('[ckpt] MISSING under checkpoints/$subName'); sys.exit(1)" + "`n" +
          "s = int(tf.train.load_checkpoint(p).get_tensor('step/.ATTRIBUTES/VARIABLE_VALUE'))" + "`n" +
          "print('[ckpt] %s step=%d (need >= $niter)' % (p, s))" + "`n" +
          "sys.exit(0 if s >= $niter else 2)" + "`n"

    $v = Invoke-Step "verify the paper checkpoint reached step $niter" deeptex `
        @("python", "-c", $py) "verify-paper-ckpt" $ctrRoot
    if ($v.Code -ne 0) {
        throw "the paper run printed [metrics] but its checkpoint does not reach step " +
              "$niter (exit $($v.Code), see $($v.Log)). Check save_steps in " +
              "$PaperConfig and free disk."
    }
    if ($v.Text -notmatch "\[ckpt\] .*step=") {
        throw "the checkpoint verification produced no [ckpt] line - it did not " +
              "actually run (see $($v.Log))."
    }

    Say "  paper training OK -> $ckpt" "Green"
    if ($PaperRoot -eq "runs/deeptex/paper") {
        Say "  evaluate it with:  -Stage eval -EvalTag paper -Config $PaperConfig ``" "DarkGray"
        Say "                     -CkptDir $ctrRoot/checkpoints/$subName" "DarkGray"
    } else {
        Say "  evaluate it with:  -Stage eval-arms -PaperConfig $PaperConfig -PaperRoot $PaperRoot" "DarkGray"
    }
}

function Stage-Eval {
    $out = "/workspace/runs/deeptex/eval/$EvalTag"
    # The paper config is upstream's verbatim, so it has no checkpoint_root key for
    # deeptex_compare.py to derive from - that arm must pass -CkptDir explicitly.
    $ck = if ($CkptDir) { @("--ckpt-dir", $CkptDir) }
          elseif ($Config -eq $PaperConfig) {
              # The paper config is upstream's verbatim and has no checkpoint_root
              # key for deeptex_compare.py to derive from, so point at where
              # Stage-Paper put it.
              $sn = (Select-String -Path (Join-Path $repo $PaperConfig) -Pattern "^sub_name:\s*'(.+)'").Matches.Groups[1].Value
              @("--ckpt-dir", "/workspace/$PaperRoot/checkpoints/$sn")
          } else { @() }
    $r = Invoke-Step "super-resolve baseColor + render for every material" deeptex `
        (@("python","/workspace/variants/deeptex/infer/deeptex_compare.py","--cfg","/workspace/$Config",
          "--root","/workspace/runs/matsynth","--out",$out) + $ck) "eval-sr"
    if ($r.Code -ne 0) { throw "eval failed (see $($r.Log))" }

    $r = Invoke-Step "comparison sets + metrics (5 columns)" dataprep `
        @("python","variants/deeptex/paper/deeptex_postprocess.py","--all","--metrics","--deeptex-root",$out) "eval-postprocess"
    if ($r.Code -ne 0) { throw "postprocess failed (see $($r.Log))" }

    $r = Invoke-Step "montages" dataprep `
        @("python","variants/deeptex/paper/deeptex_montage.py","--all","--collect","/workspace/runs/montages") "eval-montage"
    if ($r.Code -ne 0) { throw "montage failed (see $($r.Log))" }
}

# ---------------------------------------------------------------- main

Say "TexSR / DeepTextures   stage=$Stage dataset=$Dataset   logs -> runs\_logs\$stamp" "White"
if (-not $SkipPreflight) { Invoke-DeeptexPreflight } else { Say "preflight SKIPPED" "Yellow" }

switch ($Stage) {
    "preflight" { }
    "prep"      { Stage-Prep }
    "smoke"     { Stage-Smoke }
    "paper"     { Stage-Paper }
    "eval"      { Stage-Eval }
    "all"       {
        # paper-only by decision (2026-09-18): see docs/DEEPTEX-DEVIATIONS.md 0.4.
        Stage-Prep
        Stage-Smoke
        Stage-Paper
        Stage-Eval
    }
}

Say "`ndone. logs in runs\_logs\$stamp" "Green"
