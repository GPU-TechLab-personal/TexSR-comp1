<#
.SYNOPSIS
    The render-loss arms: grouped record, gates, training, render-space evaluation.

.DESCRIPTION
    Stages moved out of scripts/deeptex/run_deeptex.ps1 with their names and
    parameters unchanged (variants/deeptex/render_loss/README.md). The trainer
    is paper_loop/main.py --experiment render (old deeptex_train.py flags, L1).

.EXAMPLE
    variants\deeptex\render_loss\run.ps1 -Stage prep-grouped
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate parity
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate calibrate
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate overfit
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate smoke
    variants\deeptex\render_loss\run.ps1 -Stage gate-render -Gate premise
    variants\deeptex\render_loss\run.ps1 -Stage render-train -Arm r1 -KeepAllCheckpoints
    variants\deeptex\render_loss\run.ps1 -Stage eval-render -Arms render_r0a,render_r0b,render_r1 -Steps 30000,35000,40000
    variants\deeptex\render_loss\run.ps1 -Stage eval-perceptual -Arms render_r0a,render_r0b,render_r1 -Steps 30000,35000,40000
#>
[CmdletBinding()]
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("preflight","prep-grouped","gate-render","render-train","eval-render","eval-perceptual")]
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
#   Stage-PrepGrouped        run_deeptex.ps1:1008-1030
#   Invoke-GateParity        run_deeptex.ps1:1032-1062
#   Invoke-GateCalibrate     run_deeptex.ps1:1064-1084
#   Invoke-GateOverfit       run_deeptex.ps1:1086-1140
#   Invoke-GateSmoke         run_deeptex.ps1:1142-1226
#   Invoke-GatePremise       run_deeptex.ps1:1228-1241
#   Stage-GateRender         run_deeptex.ps1:1243-1252
#   Stage-RenderTrain        run_deeptex.ps1:1254-1333
#   Get-DisplayName          run_deeptex.ps1:1335-1347 (now util/lib.ps1, from variants/arms.json)
#   Write-RenderLossReadme   run_deeptex.ps1:1349-1415
#   Stage-EvalRender         run_deeptex.ps1:1417-1528
#   Stage-EvalPerceptual     run_deeptex.ps1:1530-1611

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
            (@("python","/workspace/variants/deeptex/render_loss/build/texsr_build_grouped_tfrecord.py",
               "--root",$MS_PAIRS,"--out",$GROUPED) + $forceArg) "prepgr-build"
        if ($r.Code -ne 0) { throw "grouped TFRecord build failed (see $($r.Log))" }
    }
    $r = Invoke-Step "verify the grouped TFRecord (every PNG against its pair, byte for byte)" deeptex `
        @("python","/workspace/variants/deeptex/render_loss/build/texsr_verify_grouped.py",
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
        @("python","/workspace/variants/deeptex/render_loss/gates/render_gate.py","--part","export","--cfg",$cfg,
          "--eval-root",$EVAL_ROOT,"--pairs-root",$MS_PAIRS,"--out",$RGATE) "gate-parity-export"
    if ($r.Code -ne 0) {
        if (-not (Test-Path $npz)) { throw "TF export failed and wrote no parity_tf.npz (see $($r.Log))" }
        $failed += "TF export (see $($r.Log))"
    }
    $r = Invoke-Step "parity 2/3: MatUp torch renderer vs the TF port (+ light table, float64 oracle)" matup `
        @("python","/workspace/variants/deeptex/render_loss/gates/render_parity_torch.py","--part","all",
          "--npz","$RGATE/parity_tf.npz","--eval-root",$EVAL_ROOT,"--out",$RGATE) "gate-parity-torch"
    if ($r.Code -ne 0) { $failed += "torch parity (see $($r.Log))" }
    $r = Invoke-Step "parity 3/3: grads, flow, flip/rot grid, chunks, mask, graph ops" deeptex `
        @("python","/workspace/variants/deeptex/render_loss/gates/render_gate.py","--part","grads,flow,grid,chunks,mask,graph",
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
        (@("python","/workspace/variants/deeptex/render_loss/gates/render_calibrate.py","--cfg","/workspace/$R1_CFG") + $ck +
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
            @("python","/workspace/variants/deeptex/paper_loop/main.py","--experiment","render",
              "--cfg_path","/workspace/variants/deeptex/render_loss/$a.yaml",
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
        $cfgRel = "variants/deeptex/render_loss/$a.yaml"
        $sn = Read-CfgValue $cfgRel "sub_name"
        $niter = [int](Read-CfgValue $cfgRel "niter")
        $root = "$RSMOKE/smoke_$a"
        Clear-SmokeDir $root
        $wr = if ($a -eq "r1") { $w } else { "0" }
        # log every 25 so run 2 has several clean windows; the monitor cadence is
        # render-train's, so the measured rate includes it.
        $base = @("python","/workspace/variants/deeptex/paper_loop/main.py","--experiment","render","--cfg_path","/workspace/$cfgRel",
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
        @("python","/workspace/variants/deeptex/matsynth/eval_arms.py","--cfg","/workspace/$R1_CFG",
          "--ckpt-dir","$RSMOKE/smoke_r1/checkpoints/$sn","--root",$EVAL_ROOT,
          "--material","acg_concrete_026","--out",$ev) "gate-smoke-cmp"
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
        (@("python","/workspace/variants/deeptex/render_loss/eval/svbrdf_render_eval.py","--root",$EVAL_ROOT) + $armArgs +
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
    $cfgRel = "variants/deeptex/render_loss/$Arm.yaml"
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
            (@("python","/workspace/variants/deeptex/paper_loop/main.py","--experiment","render","--cfg_path","/workspace/$cfgRel",
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

function Write-RenderLossReadme([string]$dst, $ref, $rows, [int]$last, [string[]]$tags, [int[]]$stepList, $suite) {
    $renRel = $suite.Render -replace '^/workspace/', ''
    $repRel = $suite.Report -replace '^/workspace/', ''
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add('# DeepTextures + MatUp render loss - map montages and render sheets')
    $L.Add('')
    $L.Add(('Generated {0} by `variants/deeptex/render_loss/run.ps1 -Stage eval-render`. Numbers behind the verdict: ' +
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
    $L.Add(('powershell -ExecutionPolicy Bypass -File variants/deeptex/render_loss/run.ps1 -Stage eval-render -Arms {0} -Steps {1}' -f
            ($tags -join ','), ($stepList -join ',')))
    $L.Add('# then, on the same folders (perceptual_summary.md, perceptual_report.json):')
    $L.Add(('powershell -ExecutionPolicy Bypass -File variants/deeptex/render_loss/run.ps1 -Stage eval-perceptual -Arms {0} -Steps {1}' -f
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
              "-PaperConfig variants/deeptex/matsynth/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all"
    }

    # 1. whole-image SR, every map, every (arm, step)
    $rows = @(Invoke-CompareAtSteps $tags $stepList $suite.Eval "evalr")

    # 2. the predicted SVBRDFs, rendered by MatUp's own renderer - every step
    $armArgs = @("--arm", "matsynth_all=$refOut")
    foreach ($p in $rows) { $armArgs += @("--arm", "$($p.Label)=$($p.Out)") }
    $r = Invoke-Step "render the predicted SVBRDFs with MatUp's renderer ($($rows.Count + 1) DeepTextures columns)" matup `
        (@("python","/workspace/variants/deeptex/render_loss/eval/svbrdf_render_eval.py","--root",$EVAL_ROOT) + $armArgs +
         @("--out",$suite.Render)) "evalr-render"
    if ($r.Code -ne 0) { throw "svbrdf_render_eval failed (see $($r.Log))" }

    # 3. map montages and map metrics at the last step - the existing report,
    #    in the container Stage-EvalArms runs it in. Its one arm-vs-arm
    #    head-to-head is the first variants/arms.json pair present (R1 vs R0a, the
    #    paired same-seed comparison); R0a and R1 go last so the columns keep the
    #    order the committed report has.
    $atLast = @($rows | Where-Object { $_.Step -eq $last })
    $repRows = @($atLast | Where-Object { $_.Info.Tag -notmatch '^(render|smoke)_(r0a|r1)$' }) +
               @($atLast | Where-Object { $_.Info.Tag -match '^(render|smoke)_r0a$' }) +
               @($atLast | Where-Object { $_.Info.Tag -match '^(render|smoke)_r1$' })
    $repArgs = @("--arm", "matsynth_all=$refOut")
    foreach ($p in $repRows) {
        $repArgs += @("--arm", "$($p.Info.Tag)=$($p.Out)")
    }
    $r = Invoke-Step "map montages + metrics at step $last" dataprep `
        (@("python","variants/deeptex/matsynth/deeptex_arms_report.py") + $repArgs +
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
                (@("python","/workspace/variants/deeptex/matsynth/deeptex_drift.py","--cfg","/workspace/$($info.Cfg)",
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
# docker/exp/compose.yaml, the only image with flip-evaluator. smoke_* arms
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
              "-PaperConfig variants/deeptex/matsynth/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all"
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
        (@("python","/workspace/variants/deeptex/render_loss/eval/svbrdf_perceptual_eval.py","--root",$EVAL_ROOT) + $armArgs +
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
                 ('powershell -ExecutionPolicy Bypass -File variants/deeptex/render_loss/run.ps1 -Stage eval-perceptual -Arms {0} -Steps {1}' -f
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
    "prep-grouped" { Stage-PrepGrouped }
    "gate-render"  { Stage-GateRender }
    "render-train" { Stage-RenderTrain }
    "eval-render"  { Stage-EvalRender }
    "eval-perceptual" { Stage-EvalPerceptual }
}

Say "`ndone. logs in runs\_logs\$stamp" "Green"
