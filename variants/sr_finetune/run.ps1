<#
.SYNOPSIS
    Official ESRGAN (BasicSR) / SwinIR (KAIR) on MatSynth: gates, training, arm folders, evaluation.

.DESCRIPTION
    Every stage runs a container of docker/compose.yaml (srft = texsr/torchbase;
    matup and dataprep for the existing evaluators; exp of docker/exp for the
    perceptual metrics). Arm tags are variants/arms.json's; a fine-tuned arm at a
    step is <tag>_<NN>k, a zero-shot arm has no step (its manifest says step 0).

      preflight  docker reachable, no stray containers, VRAM and disk reported
      setup      build texsr/torchbase, check the official weights (weights.py --check)
      prep       link the five maps' MatSynth pairs into runs/sr_ft/pairs/{HR,LRx4} (host), then gate data
      gate       -Gate env | weights | parity | registry | data | zeroshot | smoke   (runs/sr_ft/gate/*.json)
      train      -Arm esrgan | swinir_cls | swinir_real: 40k iterations, resuming if checkpoints exist
      infer      write_arm.py for every (arm, step) not already complete -> runs/sr_ft/eval/<label>
      eval       the evaluators on those folders + the DeepTextures references, colour
                 drift per checkpoint -> runs/montages/sr_finetune/

.EXAMPLE
    variants\sr_finetune\run.ps1 -Stage gate -Gate env
    variants\sr_finetune\run.ps1 -Stage prep
    variants\sr_finetune\run.ps1 -Stage gate -Gate zeroshot
    variants\sr_finetune\run.ps1 -Stage gate -Gate smoke
    variants\sr_finetune\run.ps1 -Stage train -Arm esrgan
    variants\sr_finetune\run.ps1 -Stage infer -Arms esrgan_ft,swinir_cls_ft,swinir_real_ft -Steps 30000,35000,40000
    variants\sr_finetune\run.ps1 -Stage eval
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("preflight","setup","prep","gate","train","infer","eval")]
    [string]   $Stage,
    [ValidateSet("env","weights","parity","registry","data","zeroshot","smoke")]
    [string]   $Gate          = "env",
    # -Stage train: which trainer (the fine-tuned arm is <Arm>_ft).
    [ValidateSet("esrgan","swinir_cls","swinir_real")]
    [string]   $Arm           = "esrgan",
    # Commas or spaces: powershell -File hands "a,b" over as ONE string.
    [string]   $Arms          = "esrgan_zs,esrgan_ft,swinir_cls_zs,swinir_cls_ft,swinir_real_zs,swinir_real_ft,swinir_realL_zs",
    # Fine-tuned arms only; zero-shot arms have one weight and ignore it.
    [string]   $Steps         = "30000,35000,40000",
    [switch]   $Force,
    [switch]   $SkipPreflight,
    [string[]] $AllowContainers = @()
)

$ErrorActionPreference = "Stop"
# variants\sr_finetune\run.ps1 -> two levels up
$repo       = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$compose    = Join-Path $repo "docker\compose.yaml"
$expCompose = Join-Path $repo "docker\exp\compose.yaml"
$stamp      = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir     = Join-Path $repo "runs\_logs\$stamp"
# ABSOLUTE container paths: the matup service starts inside its submodule
# (docker/compose.yaml working_dir), so a relative output would dirty external/.
$EVAL_ROOT  = "/workspace/runs/matsynth"
$SR         = "/workspace/runs/sr_ft"
$RGATE      = "$SR/gate"          # Show-GateResults reads this name
$SMOKE      = "/workspace/runs/variants_verify/sr_ft_smoke"
$MS_PAIRS   = "/workspace/runs/deeptex/datasets/matsynth"
# Everything -Stage infer / eval writes goes under $script:Base; the smoke gate
# points it at $SMOKE (and the checkpoints at its own runs) so it cannot touch
# runs/sr_ft or runs/montages/sr_finetune.
$script:Base      = $SR
$script:CkptRoots = @{}
# The DeepTextures columns every SR arm is compared with, at their last step.
$DT_REFS    = [ordered]@{ matsynth_all = "/workspace/runs/deeptex/eval/matsynth_all_40k"
                          render_r1    = "/workspace/runs/deeptex/eval/render_r1_40k" }
$DT_STEP    = 40000
# The material whose per-checkpoint sheet drift.py draws (the DeepTextures drift sheets use it too).
$DRIFT_SHEET = "0844_syntetic_foam"
# Row order everywhere: each model's zero-shot row above its fine-tuned one.
$ORDER      = @("esrgan_zs","esrgan_ft","swinir_cls_zs","swinir_cls_ft","swinir_real_zs","swinir_real_ft","swinir_realL_zs")
# Where each trainer keeps a step's inference weights, inside the arm's
# checkpoint_root (arms.json). Same patterns as nets.ARMS[...]["ft_file"];
# write_arm.py refuses a file whose name does not match its own copy.
$FT_FILE    = @{ esrgan_ft = "net_g_{0}.pth"; swinir_cls_ft = "{0}_E.pth"; swinir_real_ft = "{0}_E.pth" }
$TOTAL_ITER = 40000
# Each trainer's own VRAM peak in MiB, measured by gate smoke on the RTX 3080 Ti
# (2026-11-02: nvidia-smi peak minus the idle 416 MiB). -Stage train needs this
# much free plus 500 MiB: on Windows a card that runs out spills into shared
# memory and a run crawls (~30 s/iter measured) instead of failing.
$TRAIN_PEAK = @{ esrgan = 10900; swinir_cls = 7500; swinir_real = 9700 }

. (Join-Path $repo "variants\util\lib.ps1")
$script:StepNamePrefix = "texsr-srft"

function Test-ZeroShot([string]$tag) { return $tag -like "*_zs" }

function Get-SortedTags([string]$list) {
    $tags = @(Split-List $list)
    foreach ($t in $tags) {
        if ($ORDER -notcontains $t) { throw "unknown arm '$t'. Known: $($ORDER -join ', ')" }
    }
    return @($ORDER | Where-Object { $tags -contains $_ })
}

function Get-CkptRoot([string]$tag) {
    if ($script:CkptRoots[$tag]) { return $script:CkptRoots[$tag] }
    return (Get-ArmRegistry).arms.$tag.checkpoint_root
}

# The checkpoint file of a fine-tuned arm at a step: host path, container path.
# No container needed - the step is in the file name (net_g_<N>.pth, <N>_E.pth).
function Get-TorchCheckpoint([string]$tag, [int]$step) {
    $ctr = "$(Get-CkptRoot $tag)/" + ($FT_FILE[$tag] -f $step)
    return @{ Ctr = $ctr; Host = (Get-HostPath $ctr) }
}

# One row per arm folder the evaluators read: zero-shot once (step 0), a
# fine-tuned arm once per step. Out = container path of its folder.
function Get-ArmRows([string[]]$tags, [int[]]$stepList) {
    $rows = @()
    foreach ($t in $tags) {
        if (Test-ZeroShot $t) {
            $rows += @{ Tag = $t; Step = 0; Label = $t; Out = "$($script:Base)/eval/$t" }
        } else {
            foreach ($s in $stepList) {
                $label = "${t}_$(Get-StepTag $s)"
                $rows += @{ Tag = $t; Step = $s; Label = $label; Out = "$($script:Base)/eval/$label" }
            }
        }
    }
    return $rows
}

function Get-FreeVram {
    $used  = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" | Select-Object -First 1
    $total = (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) -split "`n" | Select-Object -First 1
    return @{ Free = [int]$total - [int]$used; Total = [int]$total; Used = [int]$used }
}

function Stage-Preflight {
    Say "preflight" "Cyan"
    Invoke-SharedPreflight -Repo $repo -AllowContainers $AllowContainers `
        -Reason "a stray container on the GPU makes inference and training share its memory."
    $v = Get-FreeVram
    Say "  vram           $($v.Free) MiB free of $($v.Total) MiB" "DarkGray"
    $freeGb = [math]::Round((Get-PSDrive -Name ($repo[0])).Free / 1GB, 1)
    Say "  disk           $freeGb GB free on $($repo[0]):" "DarkGray"
    if ($freeGb -lt 15) { throw "only $freeGb GB free; a training run saves ~200 MB every 5k steps per arm" }
}

function Stage-Setup {
    Say "`n-- build texsr/torchbase (docker/torchbase)" "Cyan"
    docker compose -f $compose build torchbase
    if ($LASTEXITCODE -ne 0) { throw "torchbase build failed" }
    $r = Invoke-Step "check the official weights (sha256, keys)" srft `
        @("python","variants/sr_finetune/weights.py","--check") "setup-weights"
    if ($r.Code -ne 0) { throw "weights.py --check failed (see $($r.Log)); download what it lists as missing" }
}

# The fine-tuning pairs: hard links on the host (prep_pairs.py, stdlib), checked in
# dataprep (gate_data.py reuses matsynth_make_pairs.py's exclusion code).
function Stage-Prep {
    if (-not (Test-Path (Join-Path $repo "runs\deeptex\datasets\matsynth\pairs_manifest.json"))) {
        throw "no MatSynth pairs - build them with variants\deeptex\matsynth (matsynth_make_pairs.py) first"
    }
    $extra = @(); if ($Force) { $extra = @("--force") }
    Invoke-GateScript "prep_pairs" "variants/sr_finetune/prep_pairs.py" $extra -OnHost
    Invoke-GateScript "data" "variants/sr_finetune/gates/gate_data.py" @() -Service dataprep
    Show-GateResults @("data")
}

# The trainer command of an arm. $total and $extra shorten a run (the smoke gate):
# BasicSR takes --force_yml key=value, kair_train.py --set key=value.
function Get-TrainCmd([string]$arm, [int]$total, [string[]]$extra = @(), [string]$root = "") {
    if ($arm -eq "esrgan") {
        $cmd = @("python","-W","ignore","variants/sr_finetune/esrgan/train.py")
        if ($root) { $cmd += @("--root", "$root/basicsr") }
        $cmd += @("-opt","variants/sr_finetune/esrgan/esrgan_ft.yml","--auto_resume")
        $force = @("train:total_iter=$total") + $extra
        return $cmd + @("--force_yml") + $force
    }
    $cmd = @("python","-W","ignore","variants/sr_finetune/swinir/kair_train.py",
             "--opt","variants/sr_finetune/swinir/${arm}_ft.json","--total-iter","$total")
    if ($root) { $cmd += @("--set", "path.root=$root/kair") }
    foreach ($e in $extra) { $cmd += @("--set", $e) }
    return $cmd
}

# 40k iterations of one arm. Re-running continues from its newest checkpoint
# (BasicSR --auto_resume, KAIR's highest <N>_*.pth); a finished arm is a no-op.
function Stage-Train([string]$arm) {
    $tag = "${arm}_ft"
    $final = Get-TorchCheckpoint $tag $TOTAL_ITER
    if ((Test-Path $final.Host) -and -not $Force) {
        Say "  [skip] $tag already has its step-$TOTAL_ITER checkpoint ($($final.Host))" "DarkGray"
        return
    }
    $v = Get-FreeVram
    $need = $TRAIN_PEAK[$arm] + 500
    if ($v.Free -lt $need) {
        throw "$arm needs $need MiB of VRAM free (its measured peak + 500); $($v.Free) MiB are. " +
              "On Windows the run would spill into shared memory and crawl instead of failing."
    }
    $r = Invoke-Step "train $tag to $TOTAL_ITER iterations (resumes from its newest checkpoint)" srft `
        (Get-TrainCmd $arm $TOTAL_ITER) "train-$arm"
    if ($r.Code -ne 0) { throw "training $tag failed (exit $($r.Code), see $($r.Log))" }
    if (-not (Test-Path $final.Host)) { throw "training $tag exited 0 but $($final.Host) is missing (see $($r.Log))" }
    Say "  $tag done -> $(Split-Path -Parent $final.Host)" "Green"
}

# write_arm.py for every row whose folder is missing, incomplete, at another
# step or older than its checkpoint (Get-CompleteEval), or all of them with -Force.
function Stage-Infer([string[]]$tags, [int[]]$stepList) {
    $rows = @(Get-ArmRows $tags $stepList)
    foreach ($p in $rows) {
        $ck = @()
        if (-not (Test-ZeroShot $p.Tag)) {
            $c = Get-TorchCheckpoint $p.Tag $p.Step
            if (-not (Test-Path $c.Host)) { throw "$($p.Label): no checkpoint $($c.Host) - train it first" }
            $ck = @("--ckpt", $c.Ctr, "--step", "$($p.Step)")
        }
        if (-not $Force -and (Get-CompleteEval $p.Out $p.Step)) {
            Say "  [skip] $($p.Label) complete in $($p.Out); -Force redoes it" "DarkGray"
            continue
        }
        $r = Invoke-Step "arm folder $($p.Label)" srft `
            (@("python","variants/sr_finetune/write_arm.py","--arm",$p.Tag) + $ck + @("--out",$p.Out)) "infer-$($p.Label)"
        if ($r.Code -ne 0) { throw "write_arm.py failed for $($p.Label) (exit $($r.Code), see $($r.Log))" }
    }
    return $rows
}

function Write-SrReadme([string]$dst, $refs, $rows, [int]$last, [string[]]$tags, [int[]]$stepList) {
    $L = New-Object System.Collections.Generic.List[string]
    $L.Add('# Official ESRGAN / SwinIR on MatSynth - montages and metrics')
    $L.Add('')
    $L.Add(('Generated {0} by `variants/sr_finetune/run.ps1 -Stage eval`. Setup, deviations from the official ' +
            'recipes and known behaviour: `variants/sr_finetune/README.md`.') -f (Get-Date -Format "yyyy-MM-dd"))
    $L.Add('')
    $L.Add('## Which weights each row is')
    $L.Add('')
    $L.Add('| tag | row name | weights | step | sha256 |')
    $L.Add('|---|---|---|---|---|')
    foreach ($k in $refs.Keys) {
        $L.Add(('| `{0}` | {1} | `{2}` | {3} | (TF checkpoint) |' -f $k, (Get-DisplayName $k), $refs[$k].checkpoint, $refs[$k].step))
    }
    foreach ($p in $rows) {
        $L.Add(('| `{0}` | {1} | `{2}` | {3} | `{4}` |' -f $p.Label, (Get-DisplayName $p.Label), $p.Done.checkpoint,
                $p.Done.step, ([string]$p.Done.weight_sha).Substring(0, 12)))
    }
    $L.Add('')
    $L.Add('## Before reading')
    $L.Add('')
    $L.Add(('- Map montages and map metrics show one checkpoint per fine-tuned arm (step {0}); render and perceptual' -f $last))
    $L.Add('  metrics cover every step in the table above; `checkpoints/<arm>/` follows the colour from step 0 (the')
    $L.Add('  official weights) through every saved checkpoint.')
    $L.Add('- The 1-channel maps (roughness, metallic, height) went in as three equal channels and are the mean of the')
    $L.Add('  three output channels; how far those channels drifted apart is `channel_spread` in each `sr_manifest.json`.')
    $L.Add('- The `render` column of `<material>.png` is image SR of MatUp''s LR render; `render_<material>.png` and')
    $L.Add('  `overview_render_pred.png` render each method''s predicted SVBRDF with MatUp''s renderer.')
    $L.Add('')
    $L.Add('## Files')
    $L.Add('')
    $L.Add('| file | content |')
    $L.Add('|---|---|')
    $L.Add('| `<material>.png`, `overview_<map>.png` | map montages: LR / bilinear / MatUp / DeepTextures / SR arms / GT (deeptex_arms_report.py) |')
    $L.Add('| `summary.md`, `per_material.md`, `metrics_long.csv`, `report.json` | map-space metrics |')
    $L.Add('| `render_<material>.png` | renders of the predicted SVBRDFs; local only (.gitignore), regenerated by -Stage eval |')
    $L.Add('| `overview_render_pred.png`, `render_summary.md` | render-space metrics and overview (svbrdf_render_eval.py) |')
    $L.Add('| `perceptual_summary.md`, `perceptual_report.json` | LPIPS / FLIP on those renders (svbrdf_perceptual_eval.py) |')
    $L.Add('| `dt_only/` | the SR arms against each other and GT only (deeptex_dt_montage.py) |')
    $L.Add(('| `checkpoints/<arm>/` | colour drift per checkpoint: `drift.md` (baseColor), `drift_normal.md`, `{0}` sheets (sr_finetune/drift.py) |' -f $DRIFT_SHEET))
    $L.Add('')
    $L.Add('## Regenerate')
    $L.Add('')
    $L.Add('```powershell')
    $L.Add(('powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage infer -Arms {0} -Steps {1}' -f
            ($tags -join ','), ($stepList -join ',')))
    $L.Add(('powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage eval -Arms {0} -Steps {1}' -f
            ($tags -join ','), ($stepList -join ',')))
    $L.Add('```')
    [System.IO.File]::WriteAllText((Join-Path $dst "README.md"), (($L -join "`n") + "`n"),
                                   (New-Object System.Text.UTF8Encoding $false))
}

# The existing evaluators on the arm folders, unmodified, as --arm TAG=DIR:
# render space (every step), map space (last step), perceptual (every step),
# SR-only sheets (last step), colour drift (every checkpoint), then one folder.
function Stage-Eval([string[]]$tags, [int[]]$stepList, [string]$collect) {
    $last = [int]($stepList | Measure-Object -Maximum).Maximum
    $B = $script:Base
    $suite = @{ Render = "$B/render_eval"; Report = "$B/report"; Perceptual = "$B/perceptual"; DtOnly = "$B/dt_only"; Drift = "$B/drift" }

    # 1. every folder complete and from its checkpoint, before any container starts
    $refs = [ordered]@{}
    foreach ($k in $DT_REFS.Keys) {
        $d = Get-CompleteEval $DT_REFS[$k] $DT_STEP
        if (-not $d) { throw "$(Get-HostPath $DT_REFS[$k]) is missing or incomplete (DeepTextures reference $k at $DT_STEP)" }
        $refs[$k] = $d
    }
    $rows = @(Get-ArmRows $tags $stepList)
    foreach ($p in $rows) {
        $p.Done = Get-CompleteEval $p.Out $p.Step
        if (-not $p.Done) { throw "$(Get-HostPath $p.Out) is missing or incomplete: run -Stage infer -Arms $($p.Tag) -Steps $($p.Step)" }
    }
    $dtArgs = @()
    foreach ($k in $DT_REFS.Keys) { $dtArgs += @("--arm", "${k}_$(Get-StepTag $DT_STEP)=$($DT_REFS[$k])") }
    $allArgs = $dtArgs
    foreach ($p in $rows) { $allArgs += @("--arm", "$($p.Label)=$($p.Out)") }
    # The last step's rows, named WITHOUT a step: the map report pairs exact tags
    # (arms.first_pair), and its rows read "ESRGAN official (fine-tuned)".
    $atLast = @($rows | Where-Object { (Test-ZeroShot $_.Tag) -or $_.Step -eq $last })
    $mapArgs = @()
    foreach ($k in $DT_REFS.Keys) { $mapArgs += @("--arm", "$k=$($DT_REFS[$k])") }
    foreach ($p in $atLast) { $mapArgs += @("--arm", "$($p.Tag)=$($p.Out)") }
    $srLast = @()
    foreach ($p in $atLast) { $srLast += @("--arm", "$($p.Label)=$($p.Out)") }

    # 2. render space: MatUp's renderer on every predicted SVBRDF
    $r = Invoke-Step "render the predicted SVBRDFs ($($allArgs.Count / 2) columns)" matup `
        (@("python","/workspace/variants/deeptex/render_loss/eval/svbrdf_render_eval.py","--root",$EVAL_ROOT) + $allArgs +
         @("--out",$suite.Render)) "eval-render"
    if ($r.Code -ne 0) { throw "svbrdf_render_eval failed (exit $($r.Code), see $($r.Log))" }

    # 3. map space at the last step
    $r = Invoke-Step "map montages + metrics (step $last)" dataprep `
        (@("python","variants/deeptex/matsynth/deeptex_arms_report.py","--root",$EVAL_ROOT) + $mapArgs +
         @("--out",$suite.Report)) "eval-report"
    if ($r.Code -ne 0) { throw "deeptex_arms_report failed (exit $($r.Code), see $($r.Log))" }

    # 4. LPIPS / FLIP on the same renders
    $r = Invoke-Step "LPIPS + FLIP on the renders ($($allArgs.Count / 2) columns)" exp `
        (@("python","/workspace/variants/deeptex/render_loss/eval/svbrdf_perceptual_eval.py","--root",$EVAL_ROOT) + $allArgs +
         @("--out",$suite.Perceptual)) "eval-perceptual" "" $expCompose
    if ($r.Code -ne 0) { throw "svbrdf_perceptual_eval failed (exit $($r.Code), see $($r.Log))" }

    # 5. SR arms only, against each other and GT
    $r = Invoke-Step "SR-only sheets (step $last)" matup `
        (@("python","/workspace/variants/deeptex/render_loss/eval/deeptex_dt_montage.py","--root",$EVAL_ROOT) + $srLast +
         @("--out",$suite.DtOnly)) "eval-dtonly"
    if ($r.Code -ne 0) { throw "deeptex_dt_montage failed (exit $($r.Code), see $($r.Log))" }

    # 6. colour drift of each fine-tuned arm over ALL its checkpoints (not only -Steps),
    #    baseColor (with sheets) and normal. Redone when a checkpoint is newer than it.
    $ftTags = @($tags | Where-Object { -not (Test-ZeroShot $_) })
    foreach ($t in $ftTags) {
        $ckDir = Get-CkptRoot $t
        foreach ($mp in "baseColor", "normal") {
            $out = "$($suite.Drift)/${t}_$mp"
            $dj = Join-Path (Get-HostPath $out) "drift.json"
            $newest = Get-ChildItem (Get-HostPath $ckDir) -Filter "*.pth" -ErrorAction SilentlyContinue |
                      Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
            if (-not $Force -and (Test-Path $dj) -and $newest -and (Get-Item $dj).LastWriteTimeUtc -gt $newest.LastWriteTimeUtc) {
                Say "  [skip] $out is newer than every checkpoint; -Force redoes it" "DarkGray"
                continue
            }
            $sheet = @(); if ($mp -eq "baseColor") { $sheet = @("--sheet-material", $DRIFT_SHEET) }
            $r = Invoke-Step "colour drift per checkpoint: $t $mp" srft `
                (@("python","-W","ignore","variants/sr_finetune/drift.py","--arm",$t,"--ckpt-dir",$ckDir,
                   "--pairs","$MS_PAIRS/$mp","--eval-root",$EVAL_ROOT,"--eval-map",$mp,"--out",$out) + $sheet) "eval-drift-$t-$mp"
            if ($r.Code -ne 0) { throw "drift.py failed for $t $mp (exit $($r.Code), see $($r.Log))" }
        }
    }

    # 7. collect. Not the full-size sheets/.
    New-Item -ItemType Directory -Force -Path $collect | Out-Null
    $rep = Get-HostPath $suite.Report
    $ren = Get-HostPath $suite.Render
    $per = Get-HostPath $suite.Perceptual
    Copy-Item (Join-Path $rep "montages\*.png") $collect -Force
    Copy-Item (Join-Path $rep "overview_*.png") $collect -Force
    foreach ($f in "summary.md", "per_material.md", "metrics_long.csv", "report.json") { Copy-Item (Join-Path $rep $f) $collect -Force }
    $sheets = @(Get-ChildItem $ren -Filter "render_*.png" -File)
    if (-not $sheets) { throw "svbrdf_render_eval wrote no render_*.png into $ren" }
    $sheets | Copy-Item -Destination $collect -Force
    Copy-Item (Join-Path $ren "overview_render_pred.png") $collect -Force
    Copy-Item (Join-Path $ren "summary.md") (Join-Path $collect "render_summary.md") -Force
    foreach ($f in "perceptual_summary.md", "perceptual_report.json") { Copy-Item (Join-Path $per $f) $collect -Force }
    $dto = Join-Path $collect "dt_only"
    New-Item -ItemType Directory -Force -Path $dto | Out-Null
    Copy-Item (Join-Path (Get-HostPath $suite.DtOnly) "*") $dto -Recurse -Force
    foreach ($t in $ftTags) {
        $cd = Join-Path $collect "checkpoints\$t"
        New-Item -ItemType Directory -Force -Path $cd | Out-Null
        $bc = Get-HostPath "$($suite.Drift)/${t}_baseColor"
        Copy-Item (Join-Path $bc "drift.md") $cd -Force
        Copy-Item (Join-Path $bc "*.png") $cd -Force
        Copy-Item (Join-Path (Get-HostPath "$($suite.Drift)/${t}_normal") "drift.md") (Join-Path $cd "drift_normal.md") -Force
    }
    Write-SrReadme $collect $refs $rows $last $tags $stepList
    Say "  montages -> $collect" "Green"
    return $rows
}

# The shared gate runner: a gate script in srft (or on the host), its json under $RGATE.
function Invoke-GateScript([string]$name, [string]$script, [string[]]$extra = @(), [switch]$OnHost,
                           [string]$Service = "srft") {
    if ($OnHost) {
        Say "`n-- gate $name (host)" "Cyan"
        $env:PYTHONDONTWRITEBYTECODE = "1"
        & python -I (Join-Path $repo $script) @extra
        $code = $LASTEXITCODE
    } else {
        $code = (Invoke-Step "gate $name" $Service (@("python","-W","ignore",$script) + $extra) "gate-$name").Code
    }
    if ($code -ne 0) { throw "gate $name failed (exit $code)" }
}

# One trainer, short, twice, under $SMOKE: 200 iterations saving every 100 with the
# learning rate halved at 150, then the SAME command to 300 - which must resume at
# 200, not start over, and still run at the halved rate. Speed and the VRAM peak
# come from the first run.
function Invoke-TrainSmoke([string]$arm) {
    $tag = "${arm}_ft"
    $extra = if ($arm -eq "esrgan") {
        @("logger:save_checkpoint_freq=100", "logger:print_freq=20", "train:scheduler:milestones=[150]")
    } elseif ($arm -eq "swinir_real") {
        @("train.checkpoint_save=100", "train.checkpoint_print=20", "train.G_scheduler_milestones=[150]", "train.D_scheduler_milestones=[150]")
    } else {
        @("train.checkpoint_save=100", "train.checkpoint_print=20", "train.G_scheduler_milestones=[150]")
    }
    $idle = (Get-FreeVram).Used
    $job = Start-Job -ScriptBlock {
        while ($true) { (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n" | Select-Object -First 1; Start-Sleep -Seconds 2 }
    }
    $a = Invoke-Step "smoke ${tag}: 200 iterations" srft (Get-TrainCmd $arm 200 $extra $SMOKE) "smoke-$arm-a"
    Stop-Job $job
    $peak = (@(Receive-Job $job | ForEach-Object { [int]$_ }) | Measure-Object -Maximum).Maximum
    Remove-Job $job -Force
    if ($a.Code -ne 0) { throw "smoke $tag (first run) failed (exit $($a.Code), see $($a.Log))" }
    $b = Invoke-Step "smoke ${tag}: the same command to 300 (must resume at 200)" srft (Get-TrainCmd $arm 300 $extra $SMOKE) "smoke-$arm-b"
    if ($b.Code -ne 0) { throw "smoke $tag (resume) failed (exit $($b.Code), see $($b.Log))" }

    if ($arm -eq "esrgan") {
        $resumed = $b.Text -match "Resuming training from epoch: \d+, iter: 200\."
        $lrA = $a.Text -match "iter:\s+200, lr:\(5\.000e-05,\)"
        $lrB = $b.Text -match "iter:\s+300, lr:\(5\.000e-05,\)"
        $times = @([regex]::Matches($a.Text, "time \(data\): ([\d.]+)") | ForEach-Object { [double]$_.Groups[1].Value })
        $sPerIter = $times[-1]
    } else {
        $resumed = ($b.Text -match "Loading model for G \[[^\]]*/200_G\.pth\]") -and -not ($b.Text -match "0_G\.pth, 0_E\.pth")
        if ($arm -eq "swinir_real") { $resumed = $resumed -and ($b.Text -match "Loading model for D \[[^\]]*/200_D\.pth\]") }
        $lrA = $a.Text -match "iter:\s+200, lr:5\.000e-05>"
        $lrB = $b.Text -match "iter:\s+300, lr:5\.000e-05>"
        $ts = @([regex]::Matches($a.Text, "(\d\d:\d\d:\d\d\.\d+) : <epoch:\s*\d+, iter:\s+(100|200),") |
                ForEach-Object { [datetime]::ParseExact($_.Groups[1].Value, "HH:mm:ss.fff", $null) })
        $sPerIter = if ($ts.Count -ge 2) { ($ts[-1] - $ts[0]).TotalSeconds / 100 } else { $null }
    }
    $final = Test-Path (Get-TorchCheckpoint $tag 300).Host
    return [ordered]@{
        resumed_at_200 = [bool]$resumed; lr_halved_before_resume = [bool]$lrA; lr_halved_after_resume = [bool]$lrB
        step_300_checkpoint = [bool]$final
        s_per_iter = $sPerIter; est_40k_hours = $(if ($sPerIter) { [math]::Round($sPerIter * $TOTAL_ITER / 3600, 1) } else { $null })
        vram_peak_mib = $peak; vram_idle_mib = $idle; vram_own_mib = $peak - $idle; vram_budget_mib = $TRAIN_PEAK[$arm]
        ok = [bool]($resumed -and $lrA -and $lrB -and $final)
        logs = @($a.Log, $b.Log)
    }
}

function Stage-Gate {
    switch ($Gate) {
        "env" {
            Invoke-GateScript "env (host)"  "variants/sr_finetune/gates/gate_env.py" @("--part","host") -OnHost
            Invoke-GateScript "env (image)" "variants/sr_finetune/gates/gate_env.py" @("--part","image")
            Show-GateResults @("env_host","env_image")
        }
        "weights"  { Invoke-GateScript "weights"  "variants/sr_finetune/gates/gate_weights.py"; Show-GateResults @("weights") }
        "parity"   { Invoke-GateScript "parity"   "variants/sr_finetune/gates/gate_parity.py";  Show-GateResults @("parity") }
        "data"     { Invoke-GateScript "data" "variants/sr_finetune/gates/gate_data.py" @() -Service dataprep; Show-GateResults @("data") }
        # Needs registry_before.json, taken before arms.json gained the SR arms
        # (gate_registry.py --snapshot); it compares against that, not against git.
        "registry" { Invoke-GateScript "registry" "variants/sr_finetune/gates/gate_registry.py" -OnHost; Show-GateResults @("registry") }
        "zeroshot" {
            # The whole chain on the four zero-shot arms, before anything is trained:
            # the same folders and collection -Stage eval writes, under the gate's own name.
            $zs = @(Get-SortedTags "esrgan_zs,swinir_cls_zs,swinir_real_zs,swinir_realL_zs")
            Stage-Infer $zs @() | Out-Null
            $dst = Join-Path $repo "runs\montages\sr_finetune"
            $rows = @(Stage-Eval $zs @(0) $dst)
            $want = @("README.md","summary.md","render_summary.md","perceptual_summary.md","overview_render_pred.png","overview_baseColor.png")
            $mats = @(Get-ChildItem (Get-HostPath $EVAL_ROOT) -Directory | Where-Object { Test-Path (Join-Path $_.FullName "SVBRDF") })
            $want += @($mats | ForEach-Object { "$($_.Name).png"; "render_$($_.Name).png" })
            $missing = @($want | Where-Object { -not (Test-Path (Join-Path $dst $_)) })
            $dtOnly = @(Get-ChildItem (Join-Path $dst "dt_only") -Filter "*.png" -ErrorAction SilentlyContinue).Count
            if ($dtOnly -eq 0) { $missing += "dt_only/*.png" }
            Write-JsonFile (Join-Path (Get-HostPath $RGATE) "zeroshot.json") ([ordered]@{
                pass = ($missing.Count -eq 0); arms = $zs; collected = $dst; missing = $missing
                columns = @($rows | ForEach-Object { $_.Label }); dt_only_png = $dtOnly })
            Show-GateResults @("zeroshot")
            if ($missing) { throw "gate zeroshot: missing $($missing -join ', ')" }
        }
        "smoke" {
            # Every trainer: save, stop, resume, schedule, speed, VRAM; then its step-300
            # checkpoint through -Stage infer and -Stage eval (drift included) - all of it
            # under $SMOKE, from an empty folder, so nothing real is read or written.
            $hostSmoke = Get-HostPath $SMOKE
            if (Test-Path $hostSmoke) { Remove-Item -Recurse -Force $hostSmoke }
            $arms = @("esrgan","swinir_cls","swinir_real")
            $script:CkptRoots = @{ esrgan_ft      = "$SMOKE/basicsr/experiments/esrgan_ft/models"
                                   swinir_cls_ft  = "$SMOKE/kair/swinir_cls_ft/models"
                                   swinir_real_ft = "$SMOKE/kair/swinir_real_ft/models" }
            $train = [ordered]@{}
            foreach ($a in $arms) { $train[$a] = Invoke-TrainSmoke $a }
            $script:Base = $SMOKE
            $ft = @($arms | ForEach-Object { "${_}_ft" })
            Stage-Infer $ft @(300) | Out-Null
            $dst = Get-HostPath "$SMOKE/montages"
            Stage-Eval $ft @(300) $dst | Out-Null
            $missing = @(@("README.md","summary.md","render_summary.md","perceptual_summary.md") + @($ft | ForEach-Object {
                "checkpoints\$_\drift.md"; "checkpoints\$_\drift_normal.md" }) | Where-Object { -not (Test-Path (Join-Path $dst $_)) })
            $pass = ($missing.Count -eq 0) -and -not @($train.Values | Where-Object { -not $_.ok })
            Write-JsonFile (Join-Path (Get-HostPath $RGATE) "smoke.json") ([ordered]@{
                pass = $pass; train = $train; collected = $dst; missing = $missing })
            foreach ($a in $arms) {
                $t = $train[$a]
                Say ("  {0,-12} resume={1} lr={2}/{3} ckpt300={4}  {5:N3} s/iter (~{6} h for 40k)  VRAM {7} MiB own (budget {8})" -f
                     $a, $t.resumed_at_200, $t.lr_halved_before_resume, $t.lr_halved_after_resume, $t.step_300_checkpoint,
                     $t.s_per_iter, $t.est_40k_hours, $t.vram_own_mib, $t.vram_budget_mib) "DarkGray"
            }
            Show-GateResults @("smoke")
            if (-not $pass) { throw "gate smoke failed (see $(Get-HostPath $RGATE)\smoke.json)" }
        }
    }
}

# ---------------------------------------------------------------- main

Say "TexSR / official SR fine-tuning   stage=$Stage   logs -> runs\_logs\$stamp" "White"
if (-not $SkipPreflight) { Stage-Preflight } else { Say "preflight SKIPPED" "Yellow" }

switch ($Stage) {
    "preflight" { }
    "setup"     { Stage-Setup }
    "prep"      { Stage-Prep }
    "gate"      { Stage-Gate }
    "train"     { Stage-Train $Arm }
    "infer"     { Stage-Infer (Get-SortedTags $Arms) @(Split-List $Steps | ForEach-Object { [int]$_ }) | Out-Null }
    "eval"      { Stage-Eval (Get-SortedTags $Arms) @(Split-List $Steps | ForEach-Object { [int]$_ }) `
                             (Join-Path $repo "runs\montages\sr_finetune") | Out-Null }
}

Say "`ndone. logs in runs\_logs\$stamp" "Green"
