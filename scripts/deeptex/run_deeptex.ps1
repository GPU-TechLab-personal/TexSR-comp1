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
#>
[CmdletBinding()]
param(
    [ValidateSet("preflight","prep","smoke","paper","pretrain","train","eval","all")]
    [string]   $Stage         = "all",
    [string]   $Dataset       = "textures_db",
    [string]   $SourceDir     = "external/DeepTextures/data/textures_db",
    [string]   $Config        = "configs/deeptex/esrgan_paper.yaml",
    [string]   $PsnrConfig    = "configs/deeptex/psnr_texsr.yaml",
    [string]   $EvalTag       = "",
    [string]   $CkptDir       = "",
    [string]   $PaperConfig   = "configs/deeptex/esrgan_paper.yaml",
    [int]      $MaxSteps      = 0,
    [int]      $CropSize      = 480,
    [int]      $Step          = 240,
    [string]   $LrKernel      = "bicubic",
    [switch]   $Force,
    [switch]   $DeleteCheckpoints,
    [switch]   $SkipPreflight,
    [string[]] $AllowContainers = @()
)

$ErrorActionPreference = "Stop"
$repo    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$compose = Join-Path $repo "docker\compose.yaml"
$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir  = Join-Path $repo "runs\_logs\$stamp"
$DS      = "/workspace/runs/deeptex/datasets/$Dataset"
$TFREC   = "$DS/${Dataset}_lrx4.tfrecord"
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
    param([string]$Label, [string]$Service, [string[]]$Cmd, [string]$LogName,
          [string]$WorkDir = "")

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
        $out = docker compose -f $compose run --rm --name $name @wd $Service @Cmd 2>&1 |
               ForEach-Object { $_.ToString() } |
               ForEach-Object { Write-Host $_; $_ }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    $out | Out-File -FilePath $log -Encoding utf8
    return @{ Text = ($out -join "`n"); Code = $code; Log = $log }
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
    $paperRoot = "/workspace/runs/deeptex/paper"
    $hostRoot  = Join-Path $repo "runs\deeptex\paper"
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
        $cfgPath = "$paperRoot/esrgan_paper_maxsteps.yaml"
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

    $r = Invoke-Step "paper arm: upstream train_esrgan.py VERBATIM ($niter steps)" deeptex `
        @("python","/workspace/scripts/deeptex/run_upstream_train.py",
          "--cfg_path",$cfgPath,"--gpu","0","--log-every","100") "train-paper" $paperRoot
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
        @("python", "-c", $py) "verify-paper-ckpt" $paperRoot
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
    Say "  evaluate it with:  -Stage eval -EvalTag paper -Config $PaperConfig ``" "DarkGray"
    Say "                     -CkptDir $paperRoot/checkpoints/$subName" "DarkGray"
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
              @("--ckpt-dir", "/workspace/runs/deeptex/paper/checkpoints/$sn")
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

# ---------------------------------------------------------------- main

Say "TexSR / DeepTextures   stage=$Stage dataset=$Dataset   logs -> runs\_logs\$stamp" "White"
if (-not $SkipPreflight) { Invoke-DeeptexPreflight } else { Say "preflight SKIPPED" "Yellow" }

switch ($Stage) {
    "preflight" { }
    "prep"      { Stage-Prep }
    "smoke"     { Stage-Smoke }
    "paper"     { Stage-Paper }
    "pretrain"  { Stage-Train "psnr" }
    "train"     { Stage-Train "esrgan" }
    "eval"      { Stage-Eval }
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
