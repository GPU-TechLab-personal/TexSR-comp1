<#
.SYNOPSIS
    Drive the DeepTextures pipeline: prep -> smoke -> pretrain -> train -> eval -> compare.

.DESCRIPTION
    The DeepTextures sibling of run_matup.ps1. Same skeleton - preflight, staged
    `docker compose run`, stderr-as-ErrorRecord handling, per-stage logs under
    runs\_logs\<stamp>\, a summary table - but the predicates and the failure
    signatures are different, because the two methods fail differently:

    * MatUp dies when the GPU goes away; it uses CUDA graphs and has no CPU path.
      TENSORFLOW SILENTLY FALLS BACK TO CPU. A 3-hour run becomes a 3-week one
      that looks healthy the whole time, so "TF can see the GPU" is checked
      explicitly (and timed - presence alone is not enough).

    * MatUp's success signal is that radiance_LR / radiance_Upscaled were removed.
      Training has no such artefact, so success here is a STATE ASSERTION:
      [metrics] reached niter AND the checkpoint's own step tensor equals niter.
      Exit code alone is not trusted, for the same reason it is not trusted there.

    * MatUp's "abort after 2 consecutive failures" is meaningless for one long
      training run. It still applies to the per-material eval stage.

    * Resumption is NORMAL here. run_matup.ps1's model is skip-if-output-exists;
      training instead resumes into an existing checkpoint directory, which is
      what you want. So -Force alone will not delete checkpoints - that needs
      -DeleteCheckpoints as well, and even then the old directory is moved aside.

.EXAMPLE
    scripts\deeptex\run_deeptex.ps1 -Stage prep
    scripts\deeptex\run_deeptex.ps1 -Stage smoke
    scripts\deeptex\run_deeptex.ps1 -Stage pretrain
    scripts\deeptex\run_deeptex.ps1 -Stage train
    scripts\deeptex\run_deeptex.ps1 -Stage eval

.EXAMPLE
    # The MatSynth arm (docs/DEEPTEX-DEVIATIONS.md 0.9): same upstream trainer,
    # MatSynth data, its own working directory so its checkpoints/esrgan_40k
    # cannot collide with the textures_db one.
    scripts\deeptex\run_deeptex.ps1 -Stage prep-matsynth
    scripts\deeptex\run_deeptex.ps1 -Stage paper     -PaperConfig configs/deeptex/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all
    scripts\deeptex\run_deeptex.ps1 -Stage eval-arms -PaperConfig configs/deeptex/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all

.EXAMPLE
    # The render-loss study (results: runs/montages/deeptex_render_loss/): the same trainer on
    # whole materials (one crop shared by the five maps), plus MatUp's render-space
    # l2,1 as an extra loss term. The gates take minutes each, in this order;
    # calibrate only PRINTS w_render - write it into the r1 config yourself.
    scripts\deeptex\run_deeptex.ps1 -Stage prep-grouped
    scripts\deeptex\run_deeptex.ps1 -Stage gate-render -Gate parity
    scripts\deeptex\run_deeptex.ps1 -Stage gate-render -Gate calibrate
    scripts\deeptex\run_deeptex.ps1 -Stage gate-render -Gate overfit
    scripts\deeptex\run_deeptex.ps1 -Stage gate-render -Gate smoke
    scripts\deeptex\run_deeptex.ps1 -Stage gate-render -Gate premise
    # Hours each, one GPU, one after another:
    scripts\deeptex\run_deeptex.ps1 -Stage render-train -Arm r0a -KeepAllCheckpoints
    scripts\deeptex\run_deeptex.ps1 -Stage render-train -Arm r1  -KeepAllCheckpoints
    scripts\deeptex\run_deeptex.ps1 -Stage render-train -Arm r0b -KeepAllCheckpoints
    # SR per (arm, step) -> MatUp-rendered metrics and sheets -> map montages ->
    # per-checkpoint drift -> runs\montages\deeptex_render_loss\
    scripts\deeptex\run_deeptex.ps1 -Stage eval-render -Arms render_r0a,render_r0b,render_r1 -Steps 30000,35000,40000
    # End-to-end dry run on the smoke checkpoints (outputs stay in runs\deeptex\_smoke_render\):
    scripts\deeptex\run_deeptex.ps1 -Stage eval-render -Arms smoke_r1 -Steps 200
    # LPIPS + FLIP on the same renders (svbrdf_perceptual_eval.py, exp image of
    # experiments/docker/compose.yaml). Reuses the eval\<arm>_<k>k folders eval-render
    # wrote and super-resolves nothing; a missing or wrong-step folder stops it.
    # -> runs\deeptex\render_eval\perceptual\, summary + report copied to
    #    runs\montages\deeptex_render_loss\
    scripts\deeptex\run_deeptex.ps1 -Stage eval-perceptual -Arms render_r0a,render_r0b,render_r1 -Steps 30000,35000,40000
    # Beside a training run: keep the GPU to it, and the CPU to -EvalThreads (default 4)
    # threads (the preflight refuses a running container and < 6 GB free VRAM, hence
    # -SkipPreflight).
    scripts\deeptex\run_deeptex.ps1 -Stage eval-perceptual -Arms render_r0a,render_r1 -EvalDevice cpu -SkipPreflight
#>
[CmdletBinding()]
param(
    [ValidateSet("preflight","prep","prep-matsynth","smoke","paper","pretrain","train","eval","eval-arms",
                 "prep-grouped","gate-render","render-train","eval-render","eval-perceptual","all")]
    [string]   $Stage         = "all",
    [string]   $Dataset       = "textures_db",
    [string]   $SourceDir     = "external/DeepTextures/data/textures_db",
    [string]   $Config        = "configs/deeptex/esrgan_paper.yaml",
    [string]   $PsnrConfig    = "configs/deeptex/psnr_texsr.yaml",
    [string]   $EvalTag       = "",
    [string]   $CkptDir       = "",
    [string]   $PaperConfig   = "configs/deeptex/esrgan_paper.yaml",
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
    # -Stage render-train: configs/deeptex/esrgan_matsynth_render_<Arm>.yaml.
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
$repo    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
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
$R1_CFG    = "configs/deeptex/esrgan_matsynth_render_r1.yaml"
# The material deeptex_drift.py draws its per-checkpoint sheet for - the one
# runs/montages/deeptex_checkpoints/ already uses, so the sheets line up.
$DRIFT_SHEET = "0844_syntetic_foam"
# Read before the default below fills it in: eval-arms names its output after
# -PaperRoot unless a tag was given explicitly.
$EvalTagGiven = $PSBoundParameters.ContainsKey("EvalTag")
if (-not $EvalTag) { $EvalTag = $Dataset }

. (Join-Path (Split-Path -Parent $PSScriptRoot) "common\preflight.ps1")

function Say([string]$msg, [string]$colour = "Gray") { Write-Host $msg -ForegroundColor $colour }

# ---------------------------------------------------------------- preflight

function Invoke-DeeptexPreflight {
    Say "preflight" "Cyan"
    Invoke-SharedPreflight -Repo $repo -AllowContainers $AllowContainers `
        -Reason ("a stray deeptex container holds several GB of VRAM and this run " +
                 "would OOM at an arbitrary step.")

    # set_memory_growth(True) means TF does not pre-reserve, so another process's
    # residency kills the run mid-flight instead of at startup.
    $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" |
            Select-Object -First 1
    $total = (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) -split "`n" |
             Select-Object -First 1
    $free = [int]$total - [int]$used
    Say "  vram           $free MiB free of $total MiB" "DarkGray"
    # Thresholds from our own 200-step probes on this card, not from a guess:
    # peak was 4,491 MiB for the PSNR stage (batch 16) and 3,234 MiB for ESRGAN
    # (batch 8). 6 GB leaves headroom for cuDNN workspaces and fragmentation;
    # below 8 GB is workable but worth saying out loud, because set_memory_growth
    # means another process can still grow into us mid-run.
    if ($free -lt 6000) {
        throw "only $free MiB of VRAM free. The PSNR stage peaked at 4,491 MiB and " +
              "ESRGAN at 3,234 MiB in our probes, so this is too tight."
    }
    if ($free -lt 8000) {
        Say "  ! $free MiB free - above the measured peak but with little headroom" "Yellow"
    }

    $freeGb = [math]::Round((Get-PSDrive -Name ($repo[0])).Free / 1GB, 1)
    Say "  disk           $freeGb GB free on $($repo[0]):" "DarkGray"
    if ($freeGb -lt 15) {
        throw "only $freeGb GB free. A run dying on ENOSPC inside manager.save() " +
              "at step 35,000 is the worst outcome available."
    }
}

# ---------------------------------------------------------------- run a stage

function Invoke-Step {
    # $ComposeFile: another compose file for $Service (eval-perceptual's exp
    # service); empty = docker\compose.yaml, as every other stage uses.
    param([string]$Label, [string]$Service, [string[]]$Cmd, [string]$LogName,
          [string]$WorkDir = "", [string]$ComposeFile = "")

    Say "`n-- $Label" "Cyan"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir "$LogName.log"

    # Windows PowerShell 5.1 wraps every stderr line from a native command in an
    # ErrorRecord, which $ErrorActionPreference='Stop' then throws on. TF writes
    # the cuBLAS/NUMA lines (harmless, docs/OPEN-ITEMS.md #5) to stderr on EVERY
    # invocation, so this is mandatory here rather than merely prudent.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $name = "texsr-deeptex-$LogName-$stamp"
    try {
        $wd = if ($WorkDir) { @("-w", $WorkDir) } else { @() }
        $cf = if ($ComposeFile) { $ComposeFile } else { $compose }
        $out = docker compose -f $cf run --rm --name $name @wd $Service @Cmd 2>&1 |
               ForEach-Object { $_.ToString() } |
               ForEach-Object { Write-Host $_; $_ }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    $out | Out-File -FilePath $log -Encoding utf8
    return @{ Text = ($out -join "`n"); Code = $code; Log = $log }
}

# Copy every checkpoint the trainer's `checkpoint` file lists into $dst, once.
# Upstream's CheckpointManager writes ckpt-N.data/.index BEFORE it lists ckpt-N,
# so anything listed is complete. The .index goes last, so its presence in $dst
# marks a finished copy. Saves are ~18 min apart and a checkpoint is deleted only
# three saves later, so polling every 30 s cannot miss one. Returns the names copied.
$script:CopyListedCheckpoints = {
    param([string]$src, [string]$dst)
    $state = Join-Path $src "checkpoint"
    if (-not (Test-Path $state)) { return }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    foreach ($line in Get-Content $state) {
        if ($line -notmatch '^all_model_checkpoint_paths:\s*"(.+)"') { continue }
        $name = Split-Path -Leaf $Matches[1]
        if (Test-Path (Join-Path $dst "$name.index")) { continue }
        $files = @(Get-ChildItem -Path $src -Filter "$name.*")
        if (-not ($files | Where-Object { $_.Extension -eq ".index" })) { continue }
        $files | Where-Object { $_.Extension -ne ".index" } | Copy-Item -Destination $dst -Force
        $files | Where-Object { $_.Extension -eq ".index" } | Copy-Item -Destination $dst -Force
        $name
    }
}

function Get-FailureKind([string]$text, [int]$code) {
    # Ordered: the most specific cause first, because several of these co-occur.
    if ($text -match "Skipping registering GPU devices|Could not load dynamic library 'libcudart") {
        return "cpu-fallback"   # distinct from gpu-lost: the fix is different
    }
    if ($text -match "nvml error|CUDA error: unknown error|failed to create shim task|CUDA_ERROR_|cuDNN launch failure|illegal memory access") {
        return "gpu-lost"
    }
    if ($text -match "ResourceExhaustedError|OOM when allocating tensor|ran out of memory") {
        return "oom"
    }
    if ($text -match "\[fatal\] non-finite loss") { return "nan" }
    if ($text -match "DataLossError|corrupted record|is required but could not be found") { return "data" }
    if ($text -match "\[fatal\]") { return "fatal" }
    if ($code -ne 0) { return "failed" }
    return "ok"
}

function Assert-TrainingReached([string]$text, [int]$niter, [string]$label) {
    # State assertion, not exit code - the DeepTextures analogue of MatUp judging
    # success by whether radiance_LR / radiance_Upscaled were removed.
    if ($text -notmatch "\[metrics\] step=") {
        throw "$label produced no [metrics] line at all. If it also said " +
              "'training done', the TFRecord yielded zero batches."
    }
    if ($niter -gt 0 -and $text -notmatch "\[metrics\] step=$niter\b") {
        Say "  ! $label did not report reaching step $niter" "Yellow"
    }
}

# ---------------------------------------------------------------- stages

function Stage-Prep {
    $forceArg = if ($Force) { @("--force") } else { @() }

    $r = Invoke-Step "normalise $SourceDir" dataprep `
        (@("python","scripts/deeptex/texsr_normalize.py","--in","/workspace/$SourceDir",
           "--out","$DS/normalized") + $forceArg) "prep-normalize"
    if ($r.Code -ne 0) { throw "normalise failed (see $($r.Log))" }

    $r = Invoke-Step "tile into exact x4 pairs" deeptex `
        (@("python","/workspace/scripts/deeptex/texsr_make_pairs.py","--in","$DS/normalized",
           "--out",$DS,"--crop-size","$CropSize","--step","$Step",
           "--lr-kernel",$LrKernel,"--jobs","8") + $forceArg) "prep-pairs"
    if ($r.Code -ne 0) { throw "pair generation failed (see $($r.Log))" }

    $r = Invoke-Step "build TFRecord" deeptex `
        @("python","/workspace/scripts/deeptex/texsr_build_tfrecord.py","--hr","$DS/HR",
          "--lr","$DS/LRx4","--out",$TFREC,"--pairs-manifest","$DS/pairs_manifest.json",
          "--overwrite") "prep-tfrecord"
    if ($r.Code -ne 0) { throw "TFRecord build failed (see $($r.Log))" }

    $r = Invoke-Step "verify TFRecord" deeptex `
        @("python","/workspace/scripts/deeptex/texsr_verify_tfrecord.py","--tfrecord",$TFREC) "prep-verify"
    if ($r.Code -ne 0) { throw "VERIFY FAILED - do not train on this dataset (see $($r.Log))" }
}

function Stage-Smoke {
    # Everything expensive, proven in about four minutes: config paths, CUDA, the
    # VGG19 weights baked into the image, checkpoint writes, and - the part
    # test.py could never do - the train -> eval handoff.
    $r = Invoke-Step "smoke train (20 steps)" deeptex `
        @("python","/workspace/scripts/deeptex/deeptex_train.py","--cfg_path","/workspace/$PsnrConfig",
          "--stage","psnr","--tfrecord",$TFREC,"--sub-name","smoke_psnr",
          "--val-hr","$DS/HR_val","--val-lr","$DS/LRx4_val",
          "--max-steps","20","--save-steps","10","--log-every","10",
          "--val-every","20","--val-count","4") "smoke-train"
    $kind = Get-FailureKind $r.Text $r.Code
    if ($kind -ne "ok") { throw "smoke train: $kind (see $($r.Log))" }
    Assert-TrainingReached $r.Text 20 "smoke train"

    $r = Invoke-Step "smoke eval (self-test + 2 materials)" deeptex `
        @("python","/workspace/scripts/deeptex/deeptex_sr.py","--cfg","/workspace/$PsnrConfig",
          "--ckpt-dir","/workspace/runs/deeptex/checkpoints/smoke_psnr","--self-test",
          "--in-dir","/workspace/runs/matsynth/acg_concrete_026/SVBRDF",
          "--out-dir","/workspace/runs/deeptex/eval/_smoke") "smoke-eval"
    if ($r.Code -ne 0) { throw "smoke eval failed (see $($r.Log))" }
}

function Clear-Checkpoints([string]$subName) {
    $dir = Join-Path $repo "runs\deeptex\checkpoints\$subName"
    if (-not (Test-Path $dir)) { return }
    if (-not $DeleteCheckpoints) {
        Say "  resuming into existing checkpoints at $dir" "DarkGray"
        Say "  (pass -Force -DeleteCheckpoints to start over)" "DarkGray"
        return
    }
    $bak = "$dir.bak-$stamp"
    Move-Item $dir $bak
    Say "  moved existing checkpoints aside -> $bak" "Yellow"
}

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
        (@("python","/workspace/scripts/deeptex/deeptex_train.py","--cfg_path","/workspace/$cfg",
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

# The paper arm does NOT use scripts/deeptex/deeptex_train.py. It execs
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
        $keepJob = Start-Job -ArgumentList $script:CopyListedCheckpoints.ToString(), $keepSrc, $keepDst `
            -ScriptBlock {
                param($code, $src, $dst)
                $copy = [scriptblock]::Create($code)
                while ($true) {
                    & $copy $src $dst | ForEach-Object { "copied $_" }
                    Start-Sleep -Seconds 30
                }
            }
        Say "  keeping every checkpoint -> $keepDst (background job $($keepJob.Id), every 30 s)" "DarkGray"
    }
    try {
        $r = Invoke-Step "paper arm: upstream train_esrgan.py VERBATIM ($niter steps)" deeptex `
            @("python","/workspace/scripts/deeptex/run_upstream_train.py",
              "--cfg_path",$cfgPath,"--gpu","0","--log-every","100") "train-paper" $ctrRoot
    } finally {
        if ($keepJob) {
            Stop-Job $keepJob
            Receive-Job $keepJob | ForEach-Object { Say "  [keep] $_" "DarkGray" }
            Remove-Job $keepJob -Force
            # One last pass: the final save lands moments before the trainer exits.
            & $script:CopyListedCheckpoints $keepSrc $keepDst | ForEach-Object { Say "  [keep] copied $_" "DarkGray" }
            $kept = @(Get-ChildItem -Path $keepDst -Filter "ckpt-*.index" -ErrorAction SilentlyContinue)
            Say "  [keep] $($kept.Count) checkpoint(s) in $keepDst" "DarkGray"
        }
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
        (@("python","/workspace/scripts/deeptex/deeptex_compare.py","--cfg","/workspace/$Config",
          "--root","/workspace/runs/matsynth","--out",$out) + $ck) "eval-sr"
    if ($r.Code -ne 0) { throw "eval failed (see $($r.Log))" }

    $r = Invoke-Step "comparison sets + metrics (5 columns)" dataprep `
        @("python","scripts/compare/matup_postprocess.py","--all","--metrics","--deeptex-root",$out) "eval-postprocess"
    if ($r.Code -ne 0) { throw "postprocess failed (see $($r.Log))" }

    $r = Invoke-Step "montages" dataprep `
        @("python","scripts/compare/matup_montage.py","--all","--collect","/workspace/runs/montages") "eval-montage"
    if ($r.Code -ne 0) { throw "montage failed (see $($r.Log))" }
}

# ---------------------------------------------------------------- MatSynth arm

# How many times each map's pairs go into the record. Order matters: "all" must
# keep the order the original record was built with.
#   all        the paper's mix, one of each map (20% each)
#   chratio    proportional to a map's real channel count, 3:3:1:1:1 - baseColor and
#              normal are 3-channel, the grey maps are 1 channel copied three times
#   basecolor  baseColor alone - the control for "does mixing maps cause the cast"
$script:MatsynthRecipes = @{
    "all"       = [ordered]@{ baseColor = 1; normal = 1; roughness = 1; metallic = 1; height = 1 }
    "chratio"   = [ordered]@{ baseColor = 3; normal = 3; roughness = 1; metallic = 1; height = 1 }
    "basecolor" = [ordered]@{ baseColor = 1 }
}

# Pairs from MatSynth's train split -> one TFRecord over the recipe's maps -> verify.
# The degradation check runs first and is a hard gate: it rebuilds the 12
# evaluation materials and requires pixel identity with runs/matsynth, i.e. that
# the network will be trained on exactly the degradation it is scored on.
function Stage-PrepMatsynth {
    $ms  = "/workspace/runs/deeptex/datasets/matsynth"
    $rec = "$ms/matsynth_${Recipe}_lrx4.tfrecord"
    $forceArg = if ($Force) { @("--force") } else { @() }

    $r = Invoke-Step "degradation identity vs the 12 eval materials" dataprep `
        @("python","scripts/deeptex/matsynth_make_pairs.py","--check-eval") "prepms-check"
    if ($r.Code -ne 0) { throw "degradation check FAILED - do not build the training set (see $($r.Log))" }

    $r = Invoke-Step "pairs from the MatSynth train shards" dataprep `
        (@("python","scripts/deeptex/matsynth_make_pairs.py","--out",$ms,"--jobs","4") + $forceArg) "prepms-pairs"
    if ($r.Code -ne 0) { throw "pair generation failed (see $($r.Log)); rerunning resumes" }

    # ONE record, one seeded shuffle across all its maps: upstream reads with
    # shuffle=False, so the record order is the curriculum. "all" passes no
    # --repeat at all, so its record is byte-for-byte the one that already exists.
    $mix = $script:MatsynthRecipes[$Recipe]
    $io = @()
    foreach ($m in $mix.Keys) {
        $io += @("--hr","$ms/$m/HR","--lr","$ms/$m/LRx4")
        if ($Recipe -ne "all") { $io += @("--repeat", "$($mix[$m])") }
    }
    Say "  recipe $Recipe : $(($mix.Keys | ForEach-Object { '{0} x{1}' -f $_, $mix[$_] }) -join ', ')" "DarkGray"
    $r = Invoke-Step "build one TFRecord ($Recipe)" deeptex `
        (@("python","/workspace/scripts/deeptex/texsr_build_tfrecord.py") + $io +
         @("--out",$rec,"--pairs-manifest","$ms/pairs_manifest.json","--overwrite")) "prepms-tfrecord"
    if ($r.Code -ne 0) { throw "TFRecord build failed (see $($r.Log))" }

    # LR here is area-decimated from the 4K source, not imresize_np(HR), so the
    # alignment probe reports a nonzero max as a warning by design; a small
    # sample is enough to catch a gross misalignment.
    $r = Invoke-Step "verify TFRecord" deeptex `
        @("python","/workspace/scripts/deeptex/texsr_verify_tfrecord.py","--tfrecord",$rec,
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
        @("python","/workspace/scripts/deeptex/deeptex_compare.py","--cfg","/workspace/$PaperConfig",
          "--ckpt-dir","/workspace/$PaperRoot/checkpoints/$sn","--root","/workspace/runs/matsynth",
          "--out",$armOut,"--maps","all","--scalar-collapse","mean") "evalarms-sr-$tag"
    if ($r.Code -ne 0) { throw "SR failed for $tag (see $($r.Log))" }

    # The baseline: the textures_db checkpoint on the same six inputs. Its
    # existing eval/paper holds baseColor and render only.
    $baseCfg = "configs/deeptex/esrgan_paper.yaml"
    $baseSn  = (Select-String -Path (Join-Path $repo $baseCfg) -Pattern "^sub_name:\s*'(.+)'").Matches.Groups[1].Value
    $baseOut = "/workspace/runs/deeptex/eval/paper_allmaps"
    $r = Invoke-Step "super-resolve all maps + render with the textures_db checkpoint" deeptex `
        @("python","/workspace/scripts/deeptex/deeptex_compare.py","--cfg","/workspace/$baseCfg",
          "--ckpt-dir","/workspace/runs/deeptex/paper/checkpoints/$baseSn","--root","/workspace/runs/matsynth",
          "--out",$baseOut,"--maps","all","--scalar-collapse","mean") "evalarms-sr-paper_allmaps"
    if ($r.Code -ne 0) { throw "SR failed for the textures_db baseline (see $($r.Log))" }

    $armArgs = @("--arm","textures_db=$baseOut")
    foreach ($t in $extra) { $armArgs += @("--arm","$t=/workspace/runs/deeptex/eval/$t") }
    $armArgs += @("--arm","$tag=$armOut")
    $r = Invoke-Step "arms report" dataprep `
        (@("python","scripts/deeptex/deeptex_arms_report.py") + $armArgs +
         @("--out","/workspace/runs/deeptex/report_$tag")) "evalarms-report"
    if ($r.Code -ne 0) { throw "report failed (see $($r.Log))" }
    Say "  report -> runs\deeptex\report_$tag\summary.md" "Green"
}

# ---------------------------------------------------------------- render-loss study
#
# The render-loss study (runs/montages/deeptex_render_loss/). Three long arms on the grouped record - R0a (seed 1,
# w_render 0), R1 (seed 1, w*), R0b (seed 2, w 0) - and the gates that must pass
# before any of them is started. Nothing here reuses Stage-Train: that one
# hardcodes the textures_db record, its validation tiles and
# runs\deeptex\checkpoints, and its failure advice ("lower batch_size", "try
# --losses corrected") would change the recipe the arms are compared on.

# Container path -> host path. Everything in this section is absolute under
# /workspace, so this is the only translation needed.
function Get-HostPath([string]$ctrPath) {
    return Join-Path $repo (($ctrPath -replace '^/workspace/', '') -replace '/', '\')
}

# A top-level scalar of a YAML config, quotes and a !!float tag stripped; $null if
# the key is absent. Same line-anchored reading as Stage-Train's sub_name/niter.
function Read-CfgValue([string]$cfgRel, [string]$key) {
    $m = Select-String -Path (Join-Path $repo $cfgRel) -CaseSensitive `
            -Pattern "^${key}:\s*(?:!!\w+\s+)?['""]?([^'""#\s]*)['""]?" | Select-Object -First 1
    if (-not $m) { return $null }
    return $m.Matches[0].Groups[1].Value
}

function Split-List([string]$s) {
    return @($s -split '[,\s]+' | Where-Object { $_ })
}

function Get-StepTag([int]$s) {
    if ($s % 1000 -eq 0) { return "$([int]($s / 1000))k" }
    return "$s"
}

function ConvertTo-Double([string]$s) {
    $v = 0.0
    if ([double]::TryParse($s, [Globalization.NumberStyles]::Float,
                           [Globalization.CultureInfo]::InvariantCulture, [ref]$v) -and
        -not [double]::IsNaN($v) -and -not [double]::IsInfinity($v)) { return $v }
    return $null
}

function Format-Float([double]$v) {
    return $v.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

function Get-Median([double[]]$v) {
    $s = @($v | Sort-Object)
    if ($s.Count -eq 0) { return $null }
    $mid = [int][math]::Floor($s.Count / 2)
    if ($s.Count % 2) { return $s[$mid] }
    return ($s[$mid - 1] + $s[$mid]) / 2.0
}

# No BOM: these files are read back by python too, and json.load rejects one.
# ConvertTo-Json also refuses non-string dictionary keys, so callers key by "$n".
function Write-JsonFile([string]$path, $obj) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null
    [System.IO.File]::WriteAllText($path, ($obj | ConvertTo-Json -Depth 8),
                                   (New-Object System.Text.UTF8Encoding $false))
}

# Arm tag -> config, checkpoint directories, sub_name. The tag is what names
# runs/deeptex/eval/<tag>_<k>k and the report columns. The render arms' checkpoint
# root comes from their config, because that is where deeptex_train.py wrote;
# smoke_* are the gate-render -Gate smoke checkpoints, for a dry run of eval-render.
function Get-ArmInfo([string]$tag) {
    $ckptRoot = $null
    switch -Regex ($tag) {
        '^matsynth_all$' {
            $cfg = "configs/deeptex/esrgan_matsynth.yaml"
            $ckptRoot = "/workspace/runs/deeptex/$tag/checkpoints" }
        '^matsynth_(chratio|basecolor)$' {
            $cfg = "configs/deeptex/esrgan_$tag.yaml"
            $ckptRoot = "/workspace/runs/deeptex/$tag/checkpoints" }
        '^render_(r0a|r0b|r1)$' {
            $cfg = "configs/deeptex/esrgan_matsynth_$tag.yaml" }
        '^smoke_(r0a|r1)$' {
            $cfg = "configs/deeptex/esrgan_matsynth_render_$($Matches[1]).yaml"
            $ckptRoot = "$RSMOKE/$tag/checkpoints" }
        default {
            throw "unknown arm tag '$tag'. Known: matsynth_all, matsynth_chratio, " +
                  "matsynth_basecolor, render_r0a, render_r0b, render_r1, smoke_r0a, smoke_r1."
        }
    }
    if (-not (Test-Path (Join-Path $repo $cfg))) { throw "$tag : config $cfg does not exist" }
    if (-not $ckptRoot) {
        $ckptRoot = Read-CfgValue $cfg "checkpoint_root"
        if (-not $ckptRoot) { $ckptRoot = "/workspace/runs/deeptex/$tag/checkpoints" }
    }
    if ($ckptRoot -notlike "/workspace/*") {
        throw "$tag : checkpoint_root '$ckptRoot' in $cfg is not under /workspace"
    }
    $sn = Read-CfgValue $cfg "sub_name"
    if (-not $sn) { throw "$tag : no sub_name in $cfg" }
    return [pscustomobject]@{
        Tag = $tag; Cfg = $cfg; SubName = $sn
        Ckpt = "$ckptRoot/$sn"
        # Where -KeepAllCheckpoints copies to: checkpoints -> checkpoints_all, as
        # Stage-Paper names it.
        CkptAll = "${ckptRoot}_all/$sn"
    }
}

# Every ckpt-*.index under the given container directories, keyed by the step
# tensor INSIDE it, not by its ckpt-N name. N counts saves, and any extra save
# (the final one after -MaxSteps, a resumed run) shifts it, so "ckpt-6" is not a
# safe handle for "30k". Globbed, not read from the `checkpoint` file, because
# checkpoints_all/ has none. One container start for all directories; the
# snippet is ONE string variable - see Stage-Paper for why that matters.
function Get-CheckpointSteps([string[]]$ctrDirs, [string]$LogName) {
    if (-not $ctrDirs) { throw "no checkpoint directory to read (none of the expected ones exists)" }
    $py = "import glob, os, sys" + "`n" +
          "import tensorflow as tf" + "`n" +
          "for d in sys.argv[1:]:" + "`n" +
          "    found = sorted(glob.glob(os.path.join(d, 'ckpt-*.index')))" + "`n" +
          "    if not found:" + "`n" +
          "        print('[ckpt-none] %s' % d)" + "`n" +
          "    for i in found:" + "`n" +
          "        p = i[:-len('.index')]" + "`n" +
          "        try:" + "`n" +
          "            s = int(tf.train.load_checkpoint(p).get_tensor('step/.ATTRIBUTES/VARIABLE_VALUE'))" + "`n" +
          "        except Exception as e:" + "`n" +
          "            print('[ckpt-bad] %s %s' % (p, type(e).__name__))" + "`n" +
          "            continue" + "`n" +
          "        print('[ckpt] %d %s' % (s, p))" + "`n"
    $r = Invoke-Step "read the step tensor of every checkpoint ($($ctrDirs.Count) dir(s))" deeptex `
        (@("python", "-c", $py) + $ctrDirs) $LogName
    if ($r.Code -ne 0) { throw "reading checkpoint steps failed (see $($r.Log))" }
    if ($r.Text -notmatch "\[ckpt(-none|-bad)?\] ") {
        throw "the checkpoint listing printed nothing - it did not actually run (see $($r.Log))"
    }
    $map = @{}
    foreach ($d in $ctrDirs) { $map[$d] = @{} }
    foreach ($m in [regex]::Matches($r.Text, '(?m)^\[ckpt\] (\d+) (\S+)')) {
        $p = $m.Groups[2].Value
        $d = $p.Substring(0, $p.LastIndexOf('/'))
        $s = [int]$m.Groups[1].Value
        if (-not $map.ContainsKey($d)) { continue }
        if ($map[$d].ContainsKey($s)) {
            # Kept as a list: Resolve-StepPrefix refuses this step, since either
            # prefix could be the (arm, step) the eval names.
            Say "  ! $p and $(@($map[$d][$s]) -join ' and ') both hold step $s; that step is ambiguous" "Yellow"
            $map[$d][$s] = @($map[$d][$s]) + $p
            continue
        }
        $map[$d][$s] = $p
    }
    return $map
}

# checkpoints_all first (every save survives there), then checkpoints.
function Resolve-StepPrefix($info, $map, [int]$want) {
    foreach ($d in @($info.CkptAll, $info.Ckpt)) {
        if ($map.ContainsKey($d) -and $map[$d].ContainsKey($want)) {
            $hit = @($map[$d][$want])
            if ($hit.Count -gt 1) {
                throw "$($info.Tag): step $want is held by $($hit.Count) checkpoints in $d " +
                      "($($hit -join ', ')). Move the stray one(s) aside."
            }
            return $hit[0]
        }
    }
    $have = @(foreach ($d in @($info.CkptAll, $info.Ckpt)) {
                  if ($map.ContainsKey($d)) { $map[$d].Keys } }) | Sort-Object -Unique
    throw "$($info.Tag): no checkpoint holds step $want (steps found: $($have -join ', ')) " +
          "under $($info.CkptAll) or $($info.Ckpt)"
}

# The Stage-Paper state assertion for an absolute directory and an EXACT step.
function Assert-CheckpointStep([string]$ctrDir, [int]$want, [string]$label, [string]$LogName) {
    $py = "import tensorflow as tf, sys" + "`n" +
          "p = tf.train.latest_checkpoint('$ctrDir')" + "`n" +
          "if not p:" + "`n" +
          "    print('[ckpt] MISSING under $ctrDir'); sys.exit(1)" + "`n" +
          "s = int(tf.train.load_checkpoint(p).get_tensor('step/.ATTRIBUTES/VARIABLE_VALUE'))" + "`n" +
          "print('[ckpt] %s step=%d (need == $want)' % (p, s))" + "`n" +
          "sys.exit(0 if s == $want else 2)" + "`n"
    $v = Invoke-Step "verify the $label checkpoint holds step $want" deeptex @("python", "-c", $py) $LogName
    if ($v.Code -ne 0) {
        throw "$label : the latest checkpoint under $ctrDir does not hold step $want " +
              "(exit $($v.Code), see $($v.Log))."
    }
    if ($v.Text -notmatch "\[ckpt\] .*step=") {
        throw "$label : the checkpoint verification printed no [ckpt] line - it did not " +
              "actually run (see $($v.Log))."
    }
}

# A deeptex_compare.py --out that covers every evaluation material at the wanted
# step: returns one manifest, or $null if the folder is missing, incomplete or
# older than its checkpoint. A complete folder at ANOTHER step throws - quietly
# reusing it would score the wrong checkpoint under the right name.
function Get-CompleteEval([string]$ctrDir, [int]$want) {
    $hostDir = Get-HostPath $ctrDir
    if (-not (Test-Path $hostDir)) { return $null }
    $mats = @(Get-ChildItem (Get-HostPath $EVAL_ROOT) -Directory |
              Where-Object { Test-Path (Join-Path $_.FullName "SVBRDF") })
    $first = $null
    foreach ($mat in $mats) {
        $mf = Join-Path $hostDir "$($mat.Name)\sr_manifest.json"
        if (-not (Test-Path $mf)) { return $null }
        $j = Get-Content $mf -Raw | ConvertFrom-Json
        if ($null -eq $j.step -or [int]$j.step -ne $want) {
            throw "$hostDir holds step $($j.step) ($($j.checkpoint)), not $want. " +
                  "Pass -Force to redo it from the right checkpoint."
        }
        # The same step is not the same checkpoint: a retrained arm (-Force
        # -DeleteCheckpoints) writes the same ckpt-N names with the same step
        # tensors. A manifest older than its checkpoint, or whose checkpoint is
        # gone, scored something else - not reused.
        $idx = (Get-HostPath ([string]$j.checkpoint)) + ".index"
        if (-not (Test-Path $idx)) {
            Say "  ! $mf : its checkpoint $($j.checkpoint) no longer exists; not reusing it" "Yellow"
            return $null
        }
        if ((Get-Item $mf).LastWriteTimeUtc -le (Get-Item $idx).LastWriteTimeUtc) {
            Say "  ! $mf predates $idx (retrained?); not reusing it" "Yellow"
            return $null
        }
        if (-not $first) { $first = $j }
    }
    return $first
}

# deeptex_compare.py --maps all for every (arm, step) -> <evalBase>/<tag>_<k>k,
# skipping complete ones unless -Force. Returns one row per pair, in arm-then-step
# order: Info, Step, Label, Out, Done (= a manifest: .checkpoint, .step).
function Invoke-CompareAtSteps([string[]]$tags, [int[]]$stepList, [string]$evalBase, [string]$logTag) {
    $infos = @($tags | ForEach-Object { Get-ArmInfo $_ })
    $plan = @()
    foreach ($info in $infos) {
        foreach ($s in $stepList) {
            $label = "$($info.Tag)_$(Get-StepTag $s)"
            $out = "$evalBase/$label"
            $done = if ($Force) { $null } else { Get-CompleteEval $out $s }
            if ($done) { Say "  [skip] $out already at step $s ($($done.checkpoint)); -Force redoes it" "DarkGray" }
            $plan += [pscustomobject]@{ Info = $info; Step = $s; Label = $label; Out = $out; Done = $done }
        }
    }
    $need = @($plan | Where-Object { -not $_.Done })
    if (-not $need) { return $plan }

    $dirs = @($need | ForEach-Object { $_.Info.CkptAll; $_.Info.Ckpt } | Select-Object -Unique |
              Where-Object { Test-Path (Get-HostPath $_) })
    foreach ($info in @($need | ForEach-Object { $_.Info } | Sort-Object Tag -Unique)) {
        if (-not ($dirs -contains $info.Ckpt -or $dirs -contains $info.CkptAll)) {
            throw "$($info.Tag): no checkpoints - neither $(Get-HostPath $info.CkptAll) " +
                  "nor $(Get-HostPath $info.Ckpt) exists"
        }
    }
    $map = Get-CheckpointSteps $dirs "$logTag-steps"
    foreach ($p in $need) {
        $prefix = Resolve-StepPrefix $p.Info $map $p.Step
        $r = Invoke-Step "super-resolve all maps + render: $($p.Info.Tag) at step $($p.Step)" deeptex `
            @("python","/workspace/scripts/deeptex/deeptex_compare.py","--cfg","/workspace/$($p.Info.Cfg)",
              "--ckpt",$prefix,"--root",$EVAL_ROOT,"--out",$p.Out,
              "--maps","all","--scalar-collapse","mean") "$logTag-sr-$($p.Label)"
        if ($r.Code -ne 0) { throw "SR failed for $($p.Label) (see $($r.Log))" }
        $p.Done = Get-CompleteEval $p.Out $p.Step
        if (-not $p.Done) {
            throw "deeptex_compare.py exited 0 but $(Get-HostPath $p.Out) lacks a manifest " +
                  "for some material (see $($r.Log))"
        }
    }
    return $plan
}

# Get-FailureKind for the render runs. TF's BFC allocator also logs "ran out of
# memory" for a retryable allocation, on the same line as "The caller indicates
# that this is not a failure", and the render arms are where that warning shows
# up. Those lines are dropped first, so a run that exited 0 is not called an OOM;
# a real one still says ResourceExhaustedError / OOM when allocating tensor.
# Get-FailureKind itself is left as the legacy stages use it.
function Get-RenderFailureKind([string]$text, [int]$code) {
    $kept = @($text -split "`n" | Where-Object { $_ -notmatch "The caller indicates that this is not a failure" })
    return Get-FailureKind ($kept -join "`n") $code
}

# Smoke and overfit runs must start from nothing, or "run 1" silently resumes an
# older smoke checkpoint. Only ever below runs/deeptex/_smoke_render.
function Clear-SmokeDir([string]$ctrDir) {
    if ($ctrDir -notlike "$RSMOKE/*") { throw "refusing to clear $ctrDir (not under $RSMOKE)" }
    $h = Get-HostPath $ctrDir
    if (Test-Path $h) {
        Remove-Item -Recurse -Force $h
        Say "  cleared $h" "DarkGray"
    }
}

function Read-WStar {
    $p = Join-Path (Get-HostPath $RGATE) "calibration.json"
    if (-not (Test-Path $p)) {
        throw "no $p - run -Stage gate-render -Gate calibrate first"
    }
    $cal = Get-Content $p -Raw | ConvertFrom-Json
    # render_calibrate.py writes the report before it exits 1, so a w_star can
    # sit in a calibration the script itself rejected.
    if ($cal.pass -ne $true) {
        throw "$p has pass=$($cal.pass): that calibration failed. Fix the cause and re-run " +
              "-Stage gate-render -Gate calibrate."
    }
    $w = ConvertTo-Double ([string]$cal.w_star)
    if ($null -eq $w -or $w -le 0) { throw "$p has no usable w_star (got '$($cal.w_star)')" }
    return $w
}

function Show-GateResults([string[]]$names) {
    foreach ($n in $names) {
        $p = Join-Path (Get-HostPath $RGATE) "$n.json"
        if (-not (Test-Path $p)) { Say ("  {0,-22} missing" -f "$n.json") "Yellow"; continue }
        try { $ok = (Get-Content $p -Raw | ConvertFrom-Json).pass }
        catch { Say ("  {0,-22} unreadable ({1})" -f "$n.json", $_.Exception.Message) "Yellow"; continue }
        Say ("  {0,-22} pass={1}" -f "$n.json", $ok) $(if ($ok -eq $true) { "Green" } else { "Red" })
    }
}

# Mirrors the Start-Job block in Stage-Paper (which stays as it is).
function Start-CheckpointKeeper([string]$src, [string]$dst) {
    $job = Start-Job -ArgumentList $script:CopyListedCheckpoints.ToString(), $src, $dst `
        -ScriptBlock {
            param($code, $src, $dst)
            $copy = [scriptblock]::Create($code)
            while ($true) {
                & $copy $src $dst | ForEach-Object { "copied $_" }
                Start-Sleep -Seconds 30
            }
        }
    Say "  keeping every checkpoint -> $dst (background job $($job.Id), every 30 s)" "DarkGray"
    return $job
}

function Stop-CheckpointKeeper($job, [string]$src, [string]$dst) {
    Stop-Job $job
    Receive-Job $job | ForEach-Object { Say "  [keep] $_" "DarkGray" }
    Remove-Job $job -Force
    # One last pass: the final save lands moments before the trainer exits.
    & $script:CopyListedCheckpoints $src $dst | ForEach-Object { Say "  [keep] copied $_" "DarkGray" }
    $kept = @(Get-ChildItem -Path $dst -Filter "ckpt-*.index" -ErrorAction SilentlyContinue)
    Say "  [keep] $($kept.Count) checkpoint(s) in $dst" "DarkGray"
}

# The shared preflight's 6 GB floor was measured on batch-8 RGB ESRGAN. A render
# arm's footprint is its own, so gate (g) measures it and this requires that peak
# plus 1,500 MiB: gpu_mem_mb is TF's allocator peak, which leaves out the CUDA
# context and cuDNN workspaces, and set_memory_growth lets another process grow
# into the card mid-run.
function Assert-RenderVram([string]$armName) {
    $sj = Join-Path (Get-HostPath $RGATE) "smoke.json"
    if (-not (Test-Path $sj)) {
        Say "  ! no runs\deeptex\render_gate\smoke.json: this arm's peak was never measured," "Yellow"
        Say "    so only the shared 6 GB floor applies (run -Stage gate-render -Gate smoke)" "Yellow"
        return
    }
    $sm = Get-Content $sj -Raw | ConvertFrom-Json
    # R0b is R0a's config with another seed; the smoke gate measures r0a and r1.
    $key = if ($armName -eq "r0b") { "r0a" } else { $armName }
    $peak = $sm.$key.gpu_mem_mb
    if ($null -eq $peak) { Say "  ! smoke.json has no gpu_mem_mb for $key" "Yellow"; return }
    $used = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" |
            Select-Object -First 1
    $total = (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) -split "`n" |
             Select-Object -First 1
    $free = [int]$total - [int]$used
    $need = [int]$peak + 1500
    Say "  vram (render)  $free MiB free; need $need = $key smoke peak $peak + 1500" "DarkGray"
    if ($free -lt $need) {
        throw "only $free MiB of VRAM free; the $key smoke run peaked at $peak MiB of TF " +
              "allocations alone. Free the card, or set render.light_chunks: 4 and re-measure " +
              "with -Gate smoke."
    }
    $apps = @(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
              Where-Object { $_ -match '\S' -and $_ -notmatch 'No running' })
    if ($apps) {
        Say "  ! other compute processes hold the GPU; with set_memory_growth they can grow into this run:" "Yellow"
        $apps | ForEach-Object { Say "      $_" "Yellow" }
    }
}

# One Example per material - the ten HR/LR PNGs of its five maps, byte-verbatim -
# so one crop and one flip/rot are shared by the five maps and the render branch
# sees a coherent SVBRDF. Built from the pairs -Stage prep-matsynth made.
function Stage-PrepGrouped {
    $hostRec = Get-HostPath $GROUPED
    if ((Test-Path $hostRec) -and -not $Force) {
        Say "  $hostRec exists - verifying it only (-Force rebuilds)" "DarkGray"
    } else {
        if (-not (Test-Path (Get-HostPath "$MS_PAIRS/_meta"))) {
            throw "no MatSynth pairs at $(Get-HostPath $MS_PAIRS) - run -Stage prep-matsynth first"
        }
        $forceArg = if ($Force) { @("--force") } else { @() }
        $r = Invoke-Step "build the grouped TFRecord (one Example per material)" deeptex `
            (@("python","/workspace/scripts/deeptex/texsr_build_grouped_tfrecord.py",
               "--root",$MS_PAIRS,"--out",$GROUPED) + $forceArg) "prepgr-build"
        if ($r.Code -ne 0) { throw "grouped TFRecord build failed (see $($r.Log))" }
    }
    $r = Invoke-Step "verify the grouped TFRecord (every PNG against its pair, byte for byte)" deeptex `
        @("python","/workspace/scripts/deeptex/texsr_verify_grouped.py",
          "--tfrecord",$GROUPED,"--root",$MS_PAIRS) "prepgr-verify"
    if ($r.Code -ne 0) { throw "VERIFY FAILED - do not train on this dataset (see $($r.Log))" }
    Say "  grouped record OK -> $hostRec" "Green"
}

# (a) light table, (b) renderer parity, (d) gradient/geometry gates. The TF side
# exports what its OWN training code path decodes and renders; MatUp's torch code
# then renders the same inputs (plus a float64 oracle). A failed check does not
# stop the other gates, so one run reports everything; only a missing
# parity_tf.npz stops the torch side, which has nothing to render without it.
function Invoke-GateParity {
    $cfg = "/workspace/$R1_CFG"
    # Removed here as well as in render_gate.py, which deletes it only after its
    # argument checks: an npz present below is this run's.
    $npz = Join-Path (Get-HostPath $RGATE) "parity_tf.npz"
    Remove-Item $npz -ErrorAction SilentlyContinue
    $failed = @()
    $r = Invoke-Step "parity 1/3: TF export (anchored crops, synthetic edge texels)" deeptex `
        @("python","/workspace/scripts/deeptex/render_gate.py","--part","export","--cfg",$cfg,
          "--eval-root",$EVAL_ROOT,"--pairs-root",$MS_PAIRS,"--out",$RGATE) "gate-parity-export"
    if ($r.Code -ne 0) {
        if (-not (Test-Path $npz)) { throw "TF export failed and wrote no parity_tf.npz (see $($r.Log))" }
        $failed += "TF export (see $($r.Log))"
    }
    $r = Invoke-Step "parity 2/3: MatUp torch renderer vs the TF port (+ light table, float64 oracle)" matup `
        @("python","/workspace/scripts/matup/render_parity_torch.py","--part","all",
          "--npz","$RGATE/parity_tf.npz","--eval-root",$EVAL_ROOT,"--out",$RGATE) "gate-parity-torch"
    if ($r.Code -ne 0) { $failed += "torch parity (see $($r.Log))" }
    $r = Invoke-Step "parity 3/3: grads, flow, flip/rot grid, chunks, mask, graph ops" deeptex `
        @("python","/workspace/scripts/deeptex/render_gate.py","--part","grads,flow,grid,chunks,mask,graph",
          "--cfg",$cfg,"--eval-root",$EVAL_ROOT,"--pairs-root",$MS_PAIRS,"--out",$RGATE) "gate-parity-tf"
    if ($r.Code -ne 0) { $failed += "TF render gates (see $($r.Log))" }
    Show-GateResults @("export","parity_lights","parity_torch","grads","flow","grid","chunks","mask","graph")
    if ($failed) { throw "parity gate FAILED: $($failed -join '; ')" }
    Say "  parity gate PASSED -> runs\deeptex\render_gate\" "Green"
}

# (e) w_render from gradient norms on six trained states: matsynth_all and chratio
# at 30k/35k/40k, resolved by step tensor. Prints w*; does NOT edit the config.
function Invoke-GateCalibrate {
    $infos = @("matsynth_all","matsynth_chratio" | ForEach-Object { Get-ArmInfo $_ })
    $dirs = @($infos | ForEach-Object { $_.CkptAll; $_.Ckpt } | Where-Object { Test-Path (Get-HostPath $_) })
    $map = Get-CheckpointSteps $dirs "gate-calibrate-steps"
    $ck = @()
    foreach ($info in $infos) {
        foreach ($s in 30000, 35000, 40000) { $ck += @("--ckpt", (Resolve-StepPrefix $info $map $s)) }
    }
    $r = Invoke-Step "calibrate w_render on 6 trained states (matsynth_all + chratio, 30k/35k/40k)" deeptex `
        (@("python","/workspace/scripts/deeptex/render_calibrate.py","--cfg","/workspace/$R1_CFG") + $ck +
         @("--batches","20","--seed","0","--tfrecord",$GROUPED,"--out","$RGATE/calibration.json")) "gate-calibrate"
    if ($r.Code -ne 0) { throw "calibration failed (see $($r.Log))" }
    $w = Format-Float (Read-WStar)
    Say "  w_star = $w   (runs\deeptex\render_gate\calibration.json)" "Green"
    Say "  NEXT: write it into $R1_CFG as" "Yellow"
    Say "          w_render: !!float $w" "Yellow"
    Say "        -Gate overfit and -Gate smoke read it from calibration.json; render-train" "Yellow"
    Say "        reads only the config, and refuses to start while it is null." "Yellow"
}

# (f) Does the render term drive the optimisation? R0a (w = 0) and R1 (w*) from the
# same matsynth_all 40k weights, on one fixed batch (same crop, no flip/rot), 300
# steps. Thresholds fixed before running: R1(300) <= 0.6 R1(1), R1(300) <= 0.85 R0a(300).
function Invoke-GateOverfit {
    # A re-run that throws part-way must not leave an older PASS behind.
    Remove-Item (Join-Path (Get-HostPath $RGATE) "overfit.json") -ErrorAction SilentlyContinue
    $w = Format-Float (Read-WStar)
    $pre = "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k"
    if (-not (Test-Path (Join-Path (Get-HostPath $pre) "checkpoint"))) {
        throw "no matsynth_all checkpoint at $(Get-HostPath $pre)"
    }
    $curves = @{}
    foreach ($a in "r0a", "r1") {
        $root = "$RSMOKE/overfit_$a"
        Clear-SmokeDir $root
        $wr = if ($a -eq "r1") { $w } else { "0" }
        $r = Invoke-Step "overfit $a (w_render=$wr): one fixed batch, 300 steps from matsynth_all 40k" deeptex `
            @("python","/workspace/scripts/deeptex/deeptex_train.py",
              "--cfg_path","/workspace/configs/deeptex/esrgan_matsynth_render_$a.yaml",
              "--stage","esrgan","--losses","upstream","--shuffle","off","--allow-dead-lr-schedule",
              "--pretrain-dir",$pre,"--no-val","--overfit-batches","1","--max-steps","300",
              "--log-every","25","--render-monitor-every","25","--w-render",$wr,
              "--checkpoint-root","$root/checkpoints","--log-root","$root/logs") "gate-overfit-$a"
        $kind = Get-RenderFailureKind $r.Text $r.Code
        if ($kind -ne "ok") { throw "overfit $a : $kind (see $($r.Log))" }
        $c = @{}
        foreach ($m in [regex]::Matches($r.Text, '\[render\] step=(\d+) l21=(\S+)')) {
            $c[[int]$m.Groups[1].Value] = ConvertTo-Double $m.Groups[2].Value
        }
        $curves[$a] = $c
    }
    $r1First = $curves["r1"][1]; $r1Last = $curves["r1"][300]; $r0Last = $curves["r0a"][300]
    $why = @()
    if ($null -eq $r1First -or $null -eq $r1Last -or $null -eq $r0Last) {
        $why += "a [render] value at step 1 or 300 is missing or non-finite"
    } else {
        if ($r1Last -gt 0.6 * $r1First) { $why += "R1(300) = $r1Last > 0.6 x R1(1) = $(0.6 * $r1First)" }
        if ($r1Last -gt 0.85 * $r0Last) { $why += "R1(300) = $r1Last > 0.85 x R0a(300) = $(0.85 * $r0Last)" }
    }
    $json = [ordered]@{
        pass = ($why.Count -eq 0); w_render = [double]$w
        r1_step1 = $r1First; r1_step300 = $r1Last; r0a_step300 = $r0Last
        thresholds = [ordered]@{ r1_300_over_r1_1 = 0.6; r1_300_over_r0a_300 = 0.85 }
        failures = $why; curves = [ordered]@{}
    }
    foreach ($a in "r0a", "r1") {
        $c = [ordered]@{}
        foreach ($k in ($curves[$a].Keys | Sort-Object)) { $c["$k"] = $curves[$a][$k] }
        $json.curves[$a] = $c
    }
    Write-JsonFile (Join-Path (Get-HostPath $RGATE) "overfit.json") $json
    Say "  R1 l21: step 1 = $r1First, step 300 = $r1Last;  R0a step 300 = $r0Last" "DarkGray"
    if ($why) { throw "overfit gate FAILED: $($why -join '; ') (runs\deeptex\render_gate\overfit.json)" }
    Say "  overfit gate PASSED -> runs\deeptex\render_gate\overfit.json" "Green"
}

# (g) 100 steps from scratch, then resume to 200, for R0a and R1: resume works,
# the step tensor is right, both arms start from identical weights (same seed),
# and the throughput and VRAM peak are MEASURED for render-train's preflight and
# the hand-over estimate. Then the train -> eval handoff on one material.
function Invoke-GateSmoke {
    # Nor an older VRAM peak: Assert-RenderVram trusts whatever smoke.json is there.
    Remove-Item (Join-Path (Get-HostPath $RGATE) "smoke.json") -ErrorAction SilentlyContinue
    $w = Format-Float (Read-WStar)
    $result = [ordered]@{}
    $inits = @{}
    foreach ($a in "r0a", "r1") {
        $cfgRel = "configs/deeptex/esrgan_matsynth_render_$a.yaml"
        $sn = Read-CfgValue $cfgRel "sub_name"
        $niter = [int](Read-CfgValue $cfgRel "niter")
        $root = "$RSMOKE/smoke_$a"
        Clear-SmokeDir $root
        $wr = if ($a -eq "r1") { $w } else { "0" }
        # log every 25 so run 2 has several clean windows; the monitor cadence is
        # render-train's, so the measured rate includes it.
        $base = @("python","/workspace/scripts/deeptex/deeptex_train.py","--cfg_path","/workspace/$cfgRel",
                  "--stage","esrgan","--losses","upstream","--shuffle","off","--allow-dead-lr-schedule",
                  "--pretrain-dir","none","--no-val","--log-every","25","--render-monitor-every","100",
                  "--w-render",$wr,"--checkpoint-root","$root/checkpoints","--log-root","$root/logs")

        $r = Invoke-Step "smoke $a run 1: 100 steps from scratch, saved at 100" deeptex `
            ($base + @("--max-steps","100","--save-steps","100")) "gate-smoke-$a-1"
        $kind = Get-RenderFailureKind $r.Text $r.Code
        if ($kind -ne "ok") { throw "smoke $a run 1: $kind (see $($r.Log))" }
        Assert-TrainingReached $r.Text 100 "smoke $a run 1"
        $m = [regex]::Match($r.Text, '\[init\] step=\d+ g_sha=([0-9a-fA-F]+) d_sha=([0-9a-fA-F]+)')
        if (-not $m.Success) { throw "smoke $a run 1 printed no [init] g_sha/d_sha line (see $($r.Log))" }
        $inits[$a] = "g_sha=$($m.Groups[1].Value) d_sha=$($m.Groups[2].Value)"
        if ($a -eq "r1" -and $inits["r1"] -ne $inits["r0a"]) {
            throw "R0a and R1 start from DIFFERENT weights ($($inits['r0a']) vs $($inits['r1'])). " +
                  "They must share seed and init, or R1 - R0a is not a paired comparison."
        }

        $r = Invoke-Step "smoke $a run 2: resume to 200" deeptex `
            ($base + @("--max-steps","200")) "gate-smoke-$a-2"
        $kind = Get-RenderFailureKind $r.Text $r.Code
        if ($kind -ne "ok") { throw "smoke $a run 2: $kind (see $($r.Log))" }
        if ($r.Text -notmatch "resumed from") { throw "smoke $a run 2 did not resume from run 1 (see $($r.Log))" }
        Assert-TrainingReached $r.Text 200 "smoke $a run 2"
        Assert-CheckpointStep "$root/checkpoints/$sn" 200 "smoke $a" "gate-smoke-$a-ckpt"

        # The first window holds the restore, the dataset skip and the trace, so
        # the rate is the median of the later ones. The memory peak is cumulative.
        $win = @([regex]::Matches($r.Text, '\[metrics\] step=(\d+) .*?steps_per_sec=([0-9.]+) gpu_mem_mb=(\d+)') |
                 ForEach-Object { [pscustomobject]@{ Step = [int]$_.Groups[1].Value
                                                     Sps = ConvertTo-Double $_.Groups[2].Value
                                                     Mem = [int]$_.Groups[3].Value } })
        if (-not $win) { throw "smoke $a run 2 printed no [metrics] steps_per_sec (see $($r.Log))" }
        $clean = if ($win.Count -gt 1) { @($win | Select-Object -Skip 1) } else { $win }
        $sps = Get-Median @($clean | ForEach-Object { $_.Sps } | Where-Object { $null -ne $_ -and $_ -gt 0 })
        if ($null -eq $sps) { throw "smoke $a run 2: no usable steps_per_sec (see $($r.Log))" }
        $mem = ($win | Measure-Object -Property Mem -Maximum).Maximum
        $result[$a] = [ordered]@{
            steps_per_sec = [math]::Round($sps, 3); gpu_mem_mb = [int]$mem
            est_hours_40k = [math]::Round($niter / $sps / 3600.0, 2)
            windows = @($win | ForEach-Object { [ordered]@{ step = $_.Step; steps_per_sec = $_.Sps } })
            init = $inits[$a]
        }
        Say ("  {0}: {1:N2} steps/s, peak {2} MiB (TF allocator), {3:N2} h for {4} steps" -f
             $a, $sps, $mem, $result[$a].est_hours_40k, $niter) "DarkGray"
        if ($a -eq "r1" -and $mem -gt 7500) {
            Say "  ! R1 peaked above 7,500 MiB: set render.light_chunks: 4 in $R1_CFG and rerun -Gate smoke" "Yellow"
        }
    }

    $sn = Read-CfgValue $R1_CFG "sub_name"
    $ev = "$RSMOKE/eval_smoke"
    Clear-SmokeDir $ev
    $r = Invoke-Step "train -> eval handoff: deeptex_compare on the R1 smoke checkpoint" deeptex `
        @("python","/workspace/scripts/deeptex/deeptex_compare.py","--cfg","/workspace/$R1_CFG",
          "--ckpt-dir","$RSMOKE/smoke_r1/checkpoints/$sn","--root",$EVAL_ROOT,
          "--material","acg_concrete_026","--maps","all","--scalar-collapse","mean","--out",$ev) "gate-smoke-cmp"
    if ($r.Code -ne 0) { throw "deeptex_compare on the smoke checkpoint failed (see $($r.Log))" }
    $pngs = @(Get-ChildItem (Join-Path (Get-HostPath $ev) "acg_concrete_026") -Filter "*.png" -ErrorAction SilentlyContinue)
    if ($pngs.Count -ne 6) {
        throw "deeptex_compare wrote $($pngs.Count) PNG(s) for acg_concrete_026, expected 6 (see $($r.Log))"
    }
    # Written only once everything above passed: render-train reads the peak from it.
    Write-JsonFile (Join-Path (Get-HostPath $RGATE) "smoke.json") $result
    Say "  smoke gate PASSED -> runs\deeptex\render_gate\smoke.json" "Green"
}

# (h) Premise and noise band, before any long run: what the render metrics say
# about the two existing MatSynth arms at 30k/35k/40k. If the render term would
# have nothing to fix but colour drift, report that before spending ~10 GPU hours.
function Invoke-GatePremise {
    $rows = @(Invoke-CompareAtSteps @("matsynth_all", "matsynth_chratio") @(30000, 35000, 40000) `
                                    "/workspace/runs/deeptex/eval" "gate-premise")
    $armArgs = @()
    foreach ($p in $rows) { $armArgs += @("--arm", "$($p.Label)=$($p.Out)") }
    $r = Invoke-Step "render the predicted SVBRDFs with MatUp's renderer (matsynth_all + chratio, 3 steps each)" matup `
        (@("python","/workspace/scripts/compare/svbrdf_render_eval.py","--root",$EVAL_ROOT) + $armArgs +
         @("--out","/workspace/runs/deeptex/render_eval/premise")) "gate-premise-render"
    if ($r.Code -ne 0) { throw "svbrdf_render_eval failed (see $($r.Log))" }
    Say "  premise -> runs\deeptex\render_eval\premise\summary.md" "Green"
}

function Stage-GateRender {
    if ($Gate -ne "premise" -and -not (Test-Path (Join-Path $repo $R1_CFG))) { throw "missing $R1_CFG" }
    switch ($Gate) {
        "parity"    { Invoke-GateParity }
        "calibrate" { Invoke-GateCalibrate }
        "overfit"   { Invoke-GateOverfit }
        "smoke"     { Invoke-GateSmoke }
        "premise"   { Invoke-GatePremise }
    }
}

# One long arm. -Force -DeleteCheckpoints moves the arm's checkpoints (and its
# checkpoints_all) aside; otherwise an existing directory is resumed - the grouped
# loader skips to the same record, so a resumed arm keeps its crop sequence.
function Stage-RenderTrain {
    $cfgRel = "configs/deeptex/esrgan_matsynth_render_$Arm.yaml"
    $info   = Get-ArmInfo "render_$Arm"
    $niter  = if ($MaxSteps -gt 0) { $MaxSteps } else { [int](Read-CfgValue $cfgRel "niter") }
    $wr     = Read-CfgValue $cfgRel "w_render"
    if ($null -ne $wr -and $wr -in @("null", "~", "")) {
        throw "w_render is null in $cfgRel. Run -Stage gate-render -Gate calibrate and write " +
              "w_star from runs\deeptex\render_gate\calibration.json into it (w_render: !!float <w>)."
    }
    if ($Arm -eq "r1") {
        # The overfit and smoke gates ran at calibration.json's w_star (--w-render);
        # this run reads the hand-copied config value, so they must be the same.
        $wStar = Read-WStar
        $wCfg  = ConvertTo-Double $wr
        if ($null -eq $wCfg -or [math]::Abs($wCfg - $wStar) -gt 1e-6 * $wStar) {
            throw "w_render in $cfgRel is '$wr', but the gates ran at w_star = $(Format-Float $wStar) " +
                  "(runs\deeptex\render_gate\calibration.json). Write w_render: !!float $(Format-Float $wStar)."
        }
    }
    $report = Join-Path (Split-Path -Parent (Get-HostPath $GROUPED)) "verify_report.json"
    if (-not (Test-Path $report)) { throw "no $report - run -Stage prep-grouped first" }
    if (-not $SkipPreflight) { Assert-RenderVram $Arm }

    $ckptHost = Get-HostPath $info.Ckpt
    $keepHost = Get-HostPath $info.CkptAll
    $nanHost  = "${ckptHost}_nan_debug"
    if ($Force -and $DeleteCheckpoints) {
        # checkpoints_all goes too: CopyListedCheckpoints skips a name already in
        # the destination, so a stale ckpt-1 there would hide the new run's ckpt-1.
        foreach ($d in @($ckptHost, $keepHost)) {
            if (Test-Path $d) {
                $bak = "$d.bak-$stamp"; Move-Item $d $bak
                Say "  moved existing checkpoints aside -> $bak" "Yellow"
            }
        }
    } elseif (Test-Path $ckptHost) {
        Say "  resuming into existing checkpoints at $ckptHost" "DarkGray"
        Say "  (pass -Force -DeleteCheckpoints to start over)" "DarkGray"
    } elseif ($KeepAllCheckpoints -and @(Get-ChildItem $keepHost -Filter "ckpt-*.index" -ErrorAction SilentlyContinue).Count) {
        throw "$keepHost holds checkpoints from an earlier run but $ckptHost is gone. The copier " +
              "would skip the new run's same-named checkpoints. Pass -Force -DeleteCheckpoints."
    }

    $extra = @()
    if ($MaxSteps -gt 0) { $extra += @("--max-steps","$MaxSteps") }
    $keepJob = $null
    if ($KeepAllCheckpoints) { $keepJob = Start-CheckpointKeeper $ckptHost $keepHost }
    try {
        $r = Invoke-Step "render-loss arm $Arm ($niter steps, w_render=$wr, $cfgRel)" deeptex `
            (@("python","/workspace/scripts/deeptex/deeptex_train.py","--cfg_path","/workspace/$cfgRel",
               "--stage","esrgan","--losses","upstream","--shuffle","off","--allow-dead-lr-schedule",
               "--pretrain-dir","none","--no-val","--log-every","100","--render-monitor-every","100") +
             $extra) "train-render-$Arm"
    } finally {
        if ($keepJob) { Stop-CheckpointKeeper $keepJob $ckptHost $keepHost }
    }
    # None of this advice changes the recipe: batch size and losses are what the
    # three arms are compared on.
    $kind = Get-RenderFailureKind $r.Text $r.Code
    switch ($kind) {
        "cpu-fallback" { throw "TF fell back to CPU - the run would take weeks. Check the GPU reservation. (see $($r.Log))" }
        "gpu-lost"     { throw "GPU fell off the bus. Reboot before retrying (docs/OPEN-ITEMS.md #10). (see $($r.Log))" }
        "oom"          { throw "out of memory. Set render.light_chunks: 4 in $cfgRel (same loss, the lights in " +
                               "recomputed chunks) and resume. Do NOT lower batch_size or change the losses. (see $($r.Log))" }
        "nan"          { throw "non-finite loss or gradient. The arm's own checkpoints were not touched; the pre-NaN " +
                               "state and the offending batch (batch_step<N>.npz) are in $nanHost. (see $($r.Log))" }
        "data"         { throw "grouped TFRecord problem - re-run -Stage prep-grouped. (see $($r.Log))" }
        "ok"           { Assert-TrainingReached $r.Text $niter "render-$Arm training" }
        default        { throw "render-$Arm training: $kind (see $($r.Log))" }
    }
    if ($r.Text -match "\[\*\] resumed from \S+ at step (\d+)") {
        Say "  ! this run RESUMED at step $($Matches[1]) - the R0a/R1 pairing no longer holds; note it with the results" "Yellow"
    }
    Assert-CheckpointStep $info.Ckpt $niter "render-$Arm" "verify-render-$Arm-ckpt"
    Say "  render-$Arm training OK -> $ckptHost" "Green"
    Say "  once all arms are done:  -Stage eval-render -Arms $Arms -Steps $Steps" "DarkGray"
}

# Row names the render sheets use (svbrdf_render_eval.py's table), for the README.
function Get-DisplayName([string]$label) {
    $names = @{ matsynth_all = "DeepTextures (matsynth_all)"
                matsynth_chratio = "DeepTextures (matsynth_chratio)"
                matsynth_basecolor = "DeepTextures (matsynth_basecolor)"
                render_r0a = "DeepTextures grouped (R0a)"; render_r0b = "DeepTextures grouped (R0b)"
                render_r1 = "DeepTextures + render loss (R1)" }
    if ($names.ContainsKey($label)) { return $names[$label] }
    if ($label -match '^(.+)_(\d+k)$' -and $names.ContainsKey($Matches[1])) {
        return "$($names[$Matches[1]]) $($Matches[2])"
    }
    return $label
}

function Write-RenderLossReadme([string]$dst, $ref, $rows, [int]$last, [string[]]$tags, [int[]]$stepList, $suite) {
    $renRel = $suite.Render -replace '^/workspace/', ''
    $repRel = $suite.Report -replace '^/workspace/', ''
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add('# DeepTextures + MatUp render loss - map montages and render sheets')
    $L.Add('')
    $L.Add(('Generated {0} by `run_deeptex.ps1 -Stage eval-render`. Numbers behind the verdict: ' +
            '`render_summary.md` (render accuracy) and `perceptual_summary.md` (LPIPS / FLIP).') -f (Get-Date -Format "yyyy-MM-dd"))
    $L.Add('')
    $L.Add('## Which checkpoint each row is')
    $L.Add('')
    $L.Add('| tag | row in render sheets | row in map montages | checkpoint | step |')
    $L.Add('|---|---|---|---|---|')
    $L.Add(('| `matsynth_all` | {0} | DeepTextures (matsynth_all) | `{1}` | {2} |' -f
            (Get-DisplayName "matsynth_all"), $ref.checkpoint, $ref.step))
    foreach ($p in $rows) {
        $mont = if ($p.Step -eq $last) { "DeepTextures ($($p.Info.Tag))" } else { "-" }
        $L.Add(('| `{0}` | {1} | {2} | `{3}` | {4} |' -f
                $p.Label, (Get-DisplayName $p.Label), $mont, $p.Done.checkpoint, $p.Done.step))
    }
    $L.Add('')
    $L.Add('## Two different "render"s')
    $L.Add('')
    $L.Add('- The `render` column of `<material>.png` and `overview_render.png` (deeptex_arms_report.py) is')
    $L.Add('  DeepTextures run as an IMAGE super-resolver on MatUp''s LR render (`render_lr.png`), exactly as')
    $L.Add('  in every earlier montage. No predicted map goes into it.')
    $L.Add('- `render_<material>.png` and `overview_render_pred.png` (svbrdf_render_eval.py) are MatUp''s own')
    $L.Add('  torch renderer applied to each method''s PREDICTED SVBRDF - all five maps - under EVAL_LIGHT and')
    $L.Add('  held-out grazing / mid / high lights, plus an EVAL_LIGHT error heatmap at the fixed gain stated')
    $L.Add('  in the sheet header. That is the quantity the render loss trains, and the R1 verdict is read')
    $L.Add('  from its numbers (`render_summary.md`).')
    $L.Add('')
    $L.Add('## Before reading')
    $L.Add('')
    $L.Add(('- The map montages show ONE checkpoint per arm (step {0}). This recipe''s global colour swings' -f $last))
    $L.Add('  from checkpoint to checkpoint (DEEPTEX-DEVIATIONS 0.10), so a single step cannot say whether')
    $L.Add('  an arm''s colour is better. The render metrics cover every step in the table above, and')
    $L.Add(('  `checkpoints/<arm>/` holds each arm''s drift tables (`drift.md` baseColor, `drift_normal.md`)' +
            ' and its per-checkpoint `{0}` sheets.') -f $DRIFT_SHEET)
    $L.Add(('- Full numbers stay outside this folder: `{0}/` (report.json, per_material.md,' -f $renRel))
    $L.Add(('  metrics_long.csv) and `{0}/` (full-size sheets/).' -f $repRel))
    $L.Add('')
    $L.Add('## Files')
    $L.Add('')
    $L.Add('| file | content |')
    $L.Add('|---|---|')
    $L.Add('| `<material>.png` | rows = LR / bilinear / MatUp / DeepTextures arms / GT, columns = five maps + render (image SR) |')
    $L.Add('| `overview_<map>.png` | rows = the evaluation materials, columns = methods |')
    $L.Add('| `summary.md`, `per_material.md`, `metrics_long.csv`, `report.json` | map-space metrics (deeptex_arms_report.py) |')
    $L.Add('| `render_<material>.png` | rows = methods, columns = renders of the predicted SVBRDF + EVAL_LIGHT error heatmap. Local only: ~13 MB each, so .gitignore keeps them out of git; eval-render regenerates them |')
    $L.Add('| `overview_render_pred.png` | rows = materials, columns = methods, EVAL_LIGHT |')
    $L.Add('| `render_summary.md` | render-space metrics (svbrdf_render_eval.py) |')
    $L.Add('| `perceptual_summary.md`, `perceptual_report.json` | LPIPS + FLIP on renders of the predicted SVBRDFs (svbrdf_perceptual_eval.py), written by `-Stage eval-perceptual`; its columns and steps are in its own header |')
    $L.Add('| `checkpoints/<arm>/` | colour drift per checkpoint (deeptex_drift.py) and its sheets |')
    $L.Add('')
    $L.Add('## Regenerate')
    $L.Add('')
    $L.Add('```powershell')
    $L.Add(('powershell -ExecutionPolicy Bypass -File scripts/deeptex/run_deeptex.ps1 -Stage eval-render -Arms {0} -Steps {1}' -f
            ($tags -join ','), ($stepList -join ',')))
    $L.Add('# then, on the same folders (perceptual_summary.md, perceptual_report.json):')
    $L.Add(('powershell -ExecutionPolicy Bypass -File scripts/deeptex/run_deeptex.ps1 -Stage eval-perceptual -Arms {0} -Steps {1}' -f
            ($tags -join ','), ($stepList -join ',')))
    $L.Add('```')
    [System.IO.File]::WriteAllText((Join-Path $dst "README.md"), (($L -join "`n") + "`n"),
                                   (New-Object System.Text.UTF8Encoding $false))
}

# Per (arm, step): whole-image SR of all maps -> MatUp-rendered metrics and sheets
# over every step -> map montages at the last step -> per-checkpoint colour drift
# -> one folder under runs\montages. smoke_* arms run the same chain but keep every
# output under runs\deeptex\_smoke_render, so a dry run cannot touch the real ones.
function Stage-EvalRender {
    $tags = @(Split-List $Arms)
    $stepList = @(Split-List $Steps | ForEach-Object { [int]$_ })
    if (-not $tags -or -not $stepList) { throw "-Arms and -Steps must not be empty" }
    $smokeTags = @($tags | Where-Object { $_ -like "smoke_*" })
    if ($smokeTags -and $smokeTags.Count -ne $tags.Count) { throw "do not mix smoke_* arms with real ones" }
    if ($smokeTags) {
        $suite = @{ Eval = "$RSMOKE/eval"; Render = "$RSMOKE/render_eval"; Report = "$RSMOKE/report"
                    Drift = "$RSMOKE/drift"; Collect = (Get-HostPath "$RSMOKE/montages") }
    } else {
        $suite = @{ Eval = "/workspace/runs/deeptex/eval"; Render = "/workspace/runs/deeptex/render_eval/main"
                    Report = "/workspace/runs/deeptex/report_render_loss"; Drift = "/workspace/runs/deeptex/drift"
                    Collect = (Join-Path $repo "runs\montages\deeptex_render_loss") }
    }
    $last = [int]($stepList | Measure-Object -Maximum).Maximum

    # The reference column: the existing matsynth_all evaluation at 40k.
    $refOut = "/workspace/runs/deeptex/eval/matsynth_all"
    $ref = Get-CompleteEval $refOut 40000
    if (-not $ref) {
        throw "runs\deeptex\eval\matsynth_all is missing, incomplete or older than its checkpoint. Run -Stage eval-arms " +
              "-PaperConfig configs/deeptex/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all"
    }

    # 1. whole-image SR, every map, every (arm, step)
    $rows = @(Invoke-CompareAtSteps $tags $stepList $suite.Eval "evalr")

    # 2. the predicted SVBRDFs, rendered by MatUp's own renderer - every step
    $armArgs = @("--arm", "matsynth_all=$refOut")
    foreach ($p in $rows) { $armArgs += @("--arm", "$($p.Label)=$($p.Out)") }
    $r = Invoke-Step "render the predicted SVBRDFs with MatUp's renderer ($($rows.Count + 1) DeepTextures columns)" matup `
        (@("python","/workspace/scripts/compare/svbrdf_render_eval.py","--root",$EVAL_ROOT) + $armArgs +
         @("--out",$suite.Render)) "evalr-render"
    if ($r.Code -ne 0) { throw "svbrdf_render_eval failed (see $($r.Log))" }

    # 3. map montages and map metrics at the last step - the existing report,
    #    in the container Stage-EvalArms runs it in. Its one arm-vs-arm
    #    head-to-head is last column vs second-to-last, so R0a and R1 go last
    #    whatever the -Arms order: R1 - R0a is the paired same-seed comparison.
    $atLast = @($rows | Where-Object { $_.Step -eq $last })
    $repRows = @($atLast | Where-Object { $_.Info.Tag -notmatch '^(render|smoke)_(r0a|r1)$' }) +
               @($atLast | Where-Object { $_.Info.Tag -match '^(render|smoke)_r0a$' }) +
               @($atLast | Where-Object { $_.Info.Tag -match '^(render|smoke)_r1$' })
    $repArgs = @("--arm", "matsynth_all=$refOut")
    foreach ($p in $repRows) {
        $repArgs += @("--arm", "$($p.Info.Tag)=$($p.Out)")
    }
    $r = Invoke-Step "map montages + metrics at step $last" dataprep `
        (@("python","scripts/deeptex/deeptex_arms_report.py") + $repArgs +
         @("--out",$suite.Report)) "evalr-report"
    if ($r.Code -ne 0) { throw "arms report failed (see $($r.Log))" }

    # 4. colour drift per checkpoint, baseColor (with sheets) and normal
    $infos = @($tags | ForEach-Object { Get-ArmInfo $_ })
    foreach ($info in $infos) {
        $src = if (@(Get-ChildItem (Get-HostPath $info.CkptAll) -Filter "ckpt-*.index" -ErrorAction SilentlyContinue).Count) {
                   $info.CkptAll } else { $info.Ckpt }
        foreach ($mp in "baseColor", "normal") {
            $out = "$($suite.Drift)/$($info.Tag)_$mp"
            $dj = Join-Path (Get-HostPath $out) "drift.json"
            if (-not $Force -and (Test-Path $dj)) {
                # Only if it is newer than every checkpoint it read: a retrained
                # arm reuses the old ckpt-N names (see Get-CompleteEval).
                $newest = Get-ChildItem (Get-HostPath $src) -Filter "ckpt-*.index" -ErrorAction SilentlyContinue |
                          Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
                if ($newest -and (Get-Item $dj).LastWriteTimeUtc -gt $newest.LastWriteTimeUtc) {
                    Say "  [skip] $out exists; -Force redoes it" "DarkGray"
                    continue
                }
                Say "  ! $out predates the newest checkpoint in $src; redoing it" "Yellow"
            }
            $sheet = if ($mp -eq "baseColor") { @("--sheet-material", $DRIFT_SHEET) } else { @() }
            $r = Invoke-Step "colour drift per checkpoint: $($info.Tag) $mp ($src)" deeptex `
                (@("python","/workspace/scripts/deeptex/deeptex_drift.py","--cfg","/workspace/$($info.Cfg)",
                   "--ckpt-dir",$src,"--pairs","$MS_PAIRS/$mp","--eval-root",$EVAL_ROOT,
                   "--eval-map",$mp,"--out",$out) + $sheet) "evalr-drift-$($info.Tag)-$mp"
            if ($r.Code -ne 0) { throw "drift failed for $($info.Tag) $mp (see $($r.Log))" }
        }
    }

    # 5. collect. Not the full-size sheets/ (hundreds of MB).
    $dst = $suite.Collect
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    $rep = Get-HostPath $suite.Report
    $ren = Get-HostPath $suite.Render
    Copy-Item (Join-Path $rep "montages\*.png") $dst -Force
    Copy-Item (Join-Path $rep "overview_*.png") $dst -Force
    foreach ($f in "summary.md", "per_material.md", "metrics_long.csv", "report.json") {
        Copy-Item (Join-Path $rep $f) $dst -Force
    }
    $sheets = @(Get-ChildItem $ren -Filter "render_*.png" -File)
    if (-not $sheets) { throw "svbrdf_render_eval wrote no render_*.png into $ren" }
    $sheets | Copy-Item -Destination $dst -Force
    Copy-Item (Join-Path $ren "overview_render_pred.png") $dst -Force
    Copy-Item (Join-Path $ren "summary.md") (Join-Path $dst "render_summary.md") -Force
    foreach ($info in $infos) {
        $cd = Join-Path $dst "checkpoints\$($info.Tag)"
        New-Item -ItemType Directory -Force -Path $cd | Out-Null
        $bc = Get-HostPath "$($suite.Drift)/$($info.Tag)_baseColor"
        Copy-Item (Join-Path $bc "drift.md") $cd -Force
        Copy-Item (Join-Path $bc "*.png") $cd -Force
        Copy-Item (Join-Path (Get-HostPath "$($suite.Drift)/$($info.Tag)_normal") "drift.md") `
                  (Join-Path $cd "drift_normal.md") -Force
    }
    Write-RenderLossReadme $dst $ref $rows $last $tags $stepList $suite
    Say "  montages -> $dst" "Green"
    Say "  render metrics -> $ren\summary.md" "Green"
}

# LPIPS + FLIP on the renders of the predicted SVBRDFs (svbrdf_perceptual_eval.py):
# is R1's lost render |grad| detail or noise? Scores the SAME eval\<arm>_<k>k folders
# eval-render scored and super-resolves nothing: a folder that is missing,
# incomplete, older than its checkpoint or at another step stops the stage
# (Get-CompleteEval, as eval-render validates them), so these numbers can only
# describe the checkpoints render_summary.md describes. Runs in the exp service of
# experiments/docker/compose.yaml, the only image with flip-evaluator. smoke_* arms
# keep everything under runs\deeptex\_smoke_render, as in Stage-EvalRender.
function Stage-EvalPerceptual {
    $tags = @(Split-List $Arms)
    $stepList = @(Split-List $Steps | ForEach-Object { [int]$_ })
    if (-not $tags -or -not $stepList) { throw "-Arms and -Steps must not be empty" }
    $smokeTags = @($tags | Where-Object { $_ -like "smoke_*" })
    if ($smokeTags -and $smokeTags.Count -ne $tags.Count) { throw "do not mix smoke_* arms with real ones" }
    if ($smokeTags) {
        $suite = @{ Eval = "$RSMOKE/eval"; Perceptual = "$RSMOKE/render_eval/perceptual"
                    Collect = (Get-HostPath "$RSMOKE/montages") }
    } else {
        $suite = @{ Eval = "/workspace/runs/deeptex/eval"; Perceptual = "/workspace/runs/deeptex/render_eval/perceptual"
                    Collect = (Join-Path $repo "runs\montages\deeptex_render_loss") }
    }

    # The reference column, exactly as Stage-EvalRender requires it.
    $refOut = "/workspace/runs/deeptex/eval/matsynth_all"
    if (-not (Get-CompleteEval $refOut 40000)) {
        throw "runs\deeptex\eval\matsynth_all is missing, incomplete or older than its checkpoint. Run -Stage eval-arms " +
              "-PaperConfig configs/deeptex/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all"
    }
    # Every folder is checked before the container starts, so a typo costs nothing.
    $armArgs = @("--arm", "matsynth_all=$refOut")
    foreach ($tag in $tags) {
        foreach ($s in $stepList) {
            $label = "${tag}_$(Get-StepTag $s)"
            $out = "$($suite.Eval)/$label"
            # A complete folder at another step throws there; its "-Force" advice
            # means eval-render's -Force, since this stage never rewrites it.
            try { $done = Get-CompleteEval $out $s }
            catch { throw "$($_.Exception.Message) (with -Stage eval-render -Arms $tag -Steps $s)" }
            if (-not $done) {
                throw "$(Get-HostPath $out) is missing, incomplete, or older than (or without) its checkpoint. " +
                      "eval-perceptual super-resolves nothing: run -Stage eval-render -Arms $tag -Steps $s first."
            }
            $armArgs += @("--arm", "$label=$out")
        }
    }
    # @(...): "auto" must add NOTHING - an array plus $null gains a $null element.
    $dev = @(switch ($EvalDevice) { "cpu" { "--device", "cpu" } "cuda" { "--device", "cuda:0" } })
    # cpu is the run-beside-training mode, so the CPU gets a cap too (-EvalThreads).
    if ($EvalDevice -eq "cpu" -and $EvalThreads -gt 0) { $dev += @("--threads", "$EvalThreads") }

    $r = Invoke-Step "LPIPS + FLIP on renders of the predicted SVBRDFs ($($armArgs.Count / 2) DeepTextures columns, device $EvalDevice)" exp `
        (@("python","/workspace/scripts/compare/svbrdf_perceptual_eval.py","--root",$EVAL_ROOT) + $armArgs +
         @("--out",$suite.Perceptual) + $dev) "evalp-perceptual" "" $expCompose
    if ($r.Code -ne 0) { throw "svbrdf_perceptual_eval failed (see $($r.Log))" }

    # Collect next to render_summary.md. Only these two: the CSV stays with the run.
    $per = Get-HostPath $suite.Perceptual
    $dst = $suite.Collect
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    foreach ($f in "perceptual_summary.md", "perceptual_report.json") {
        $src = Join-Path $per $f
        if (-not (Test-Path $src)) { throw "svbrdf_perceptual_eval exited 0 but wrote no $src (see $($r.Log))" }
        Copy-Item $src $dst -Force
    }
    # Write-RenderLossReadme lists these two files, but a README.md an earlier
    # eval-render wrote does not: add the row and this command once. No README
    # yet means eval-render has not collected here; its README will list them.
    $readme = Join-Path $dst "README.md"
    if ((Test-Path $readme) -and -not ([System.IO.File]::ReadAllText($readme)).Contains("perceptual_summary.md")) {
        $add = @('', '## Perceptual metrics', '', '| file | content |', '|---|---|',
                 '| `perceptual_summary.md`, `perceptual_report.json` | LPIPS + FLIP on renders of the predicted SVBRDFs (svbrdf_perceptual_eval.py); its columns and steps are in its own header |',
                 '', '```powershell',
                 ('powershell -ExecutionPolicy Bypass -File scripts/deeptex/run_deeptex.ps1 -Stage eval-perceptual -Arms {0} -Steps {1}' -f
                  ($tags -join ','), ($stepList -join ',')),
                 '```')
        [System.IO.File]::AppendAllText($readme, (($add -join "`n") + "`n"),
                                        (New-Object System.Text.UTF8Encoding $false))
        Say "  listed them in $readme" "DarkGray"
    }
    Say "  perceptual metrics -> $per\perceptual_summary.md" "Green"
    Say "  copied perceptual_summary.md, perceptual_report.json -> $dst" "Green"
}

# ---------------------------------------------------------------- main

Say "TexSR / DeepTextures   stage=$Stage dataset=$Dataset   logs -> runs\_logs\$stamp" "White"
if (-not $SkipPreflight) { Invoke-DeeptexPreflight } else { Say "preflight SKIPPED" "Yellow" }

switch ($Stage) {
    "preflight" { }
    "prep"      { Stage-Prep }
    "prep-matsynth" { Stage-PrepMatsynth }
    "eval-arms" { Stage-EvalArms }
    "smoke"     { Stage-Smoke }
    "paper"     { Stage-Paper }
    "pretrain"  { Stage-Train "psnr" }
    "train"     { Stage-Train "esrgan" }
    "eval"      { Stage-Eval }
    "prep-grouped" { Stage-PrepGrouped }
    "gate-render"  { Stage-GateRender }
    "render-train" { Stage-RenderTrain }
    "eval-render"  { Stage-EvalRender }
    "eval-perceptual" { Stage-EvalPerceptual }
    "all"       {
        # paper-only by decision (2026-09-18): the comparison runs against the code
        # the paper released. `pretrain` and `train` still work and are still
        # tested, but they build the corrected model, which is out of scope here.
        # See docs/DEEPTEX-DEVIATIONS.md 0.4.
        Stage-Prep
        Stage-Smoke
        Stage-Paper
        Stage-Eval
    }
}

Say "`ndone. logs in runs\_logs\$stamp" "Green"
