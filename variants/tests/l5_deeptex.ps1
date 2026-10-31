<#
.SYNOPSIS
    L5 equivalence test: scripts/deeptex/run_deeptex.ps1 (old) vs the four arm
    run.ps1 files under variants/deeptex/ (new), with docker intercepted.

.DESCRIPTION
    Every -Stage of the old orchestrator is run, in this process, once through
    the old script and once through the arm run.ps1 that now owns it
    (paper | matsynth | fixed | render_loss), with the same parameters (config
    paths in them mapped as below). Stages with several branches get one
    scenario per branch: -Force / -DeleteCheckpoints / -MaxSteps /
    -KeepAllCheckpoints, each failure signature the stage turns into advice,
    and the refusals that fire before any container starts.

    Three things are compared after the mapping below is applied to the old side:

      1. every docker argv, in order          (the contract under test)
      2. the console transcript               (stamps, sandbox root, job ids
                                               normalised)
      3. the files the script created          (relative to its repo root)

    Each scenario also carries its own expectation (how many containers, which
    message), checked on both sides, so a scenario that silently stopped
    exercising its branch FAILs instead of comparing two empty lists.

    Old -> new mapping (old side only)
    ----------------------------------
      P  substring path map, every argv element and transcript line ($PATH_SUBS)
      T  deeptex_train.py X  ->  paper_loop/main.py --experiment render|fixed X
         (render when the config is under render_loss/)
      E  deeptex_compare.py ... --maps all --scalar-collapse mean
           ->  matsynth/eval_arms.py ...   (those two are its defaults)
         deeptex_compare.py without --maps  ->  infer/deeptex_compare.py
      R  texsr_build_tfrecord.py --hr/--lr[/--repeat k] per map, --out
         .../matsynth_<R>_lrx4.tfrecord  ->  matsynth/recipes.py --recipe <R>
         --root <ms>; the old map list must equal the old ps1's recipe table
         exactly, or the call is left unmapped (and FAILs). The old
         "  recipe <R> : ..." console line is dropped: recipes.py prints it.

    How docker is intercepted
    -------------------------
    As in l5_matup.ps1: `docker` (and `nvidia-smi`) are PowerShell functions,
    which win over applications, so both orchestrators call them. The fake
    reads the container argv and leaves behind what the real script would:
    trainers print [init]/[metrics]/[render]/[*] resumed lines and write a
    ckpt-N (with its step in a ckpt-N.step sidecar) plus the `checkpoint`
    listing; the checkpoint readers (`python -c`) answer from those sidecars;
    the SR scripts write <out>/<material>/sr_manifest.json and six PNGs; every
    gate, report and eval script writes the files the stage then reads.

    Each side runs in a sandbox repo root: the script, lib.ps1 / preflight.ps1
    and the configs, byte copies at the same relative paths, under

        <main checkout>\runs\variants_verify\deeptex_l5\<stamp>\sandbox\<scenario>\<old|new>\

    Nothing is written to the real runs/ (checked at the end).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File variants\tests\l5_deeptex.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File variants\tests\l5_deeptex.ps1 -Only gate-smoke,eval-render
#>
[CmdletBinding()]
param(
    # Default: <main checkout>\runs\variants_verify\deeptex_l5
    [string]   $OutRoot = "",
    [string[]] $Only    = @()
)

$ErrorActionPreference = "Stop"
$repo  = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$t0    = Get-Date
$Only  = @($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ })

$gitCommon = (git -C $repo rev-parse --git-common-dir).Trim()
if (-not [IO.Path]::IsPathRooted($gitCommon)) { $gitCommon = Join-Path $repo $gitCommon }
$main = Split-Path -Parent (Resolve-Path $gitCommon).Path
if (-not $OutRoot) { $OutRoot = Join-Path $main "runs\variants_verify\deeptex_l5" }
$outDir = Join-Path $OutRoot $stamp
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$VD = "variants/deeptex"   # not $V: a local $v anywhere up the call stack would shadow it
$OLD_SCRIPT = "scripts\deeptex\run_deeptex.ps1"
$OLD_EXTRA  = @("scripts\common\preflight.ps1") +
              @(Get-ChildItem (Join-Path $repo "configs\deeptex") -Filter *.yaml | ForEach-Object { "configs\deeptex\$($_.Name)" })
$NEW_EXTRA  = @("variants\common\lib.ps1", "variants\common\preflight.ps1") +
              @(Get-ChildItem (Join-Path $repo "variants\deeptex") -Recurse -Filter *.yaml |
                ForEach-Object { $_.FullName.Substring($repo.Length + 1) })

# Rule P: ordered substring map (old -> new).
$PATH_SUBS = @(
    @("scripts/deeptex/texsr_normalize.py",              "$VD/paper/texsr_normalize.py"),
    @("scripts/deeptex/texsr_make_pairs.py",             "$VD/paper/texsr_make_pairs.py"),
    @("scripts/deeptex/texsr_build_tfrecord.py",         "$VD/upstream_runner/texsr_build_tfrecord.py"),
    @("scripts/deeptex/texsr_verify_tfrecord.py",        "$VD/upstream_runner/texsr_verify_tfrecord.py"),
    @("scripts/deeptex/run_upstream_train.py",           "$VD/upstream_runner/run_upstream_train.py"),
    @("scripts/deeptex/deeptex_sr.py",                   "$VD/infer/deeptex_sr.py"),
    @("scripts/compare/matup_postprocess.py",            "$VD/paper/deeptex_postprocess.py"),
    @("scripts/compare/matup_montage.py",                "$VD/paper/deeptex_montage.py"),
    @("scripts/deeptex/matsynth_make_pairs.py",          "$VD/matsynth/matsynth_make_pairs.py"),
    @("scripts/deeptex/deeptex_arms_report.py",          "$VD/matsynth/deeptex_arms_report.py"),
    @("scripts/deeptex/deeptex_drift.py",                "$VD/matsynth/deeptex_drift.py"),
    @("scripts/deeptex/texsr_build_grouped_tfrecord.py", "$VD/render_loss/build/texsr_build_grouped_tfrecord.py"),
    @("scripts/deeptex/texsr_verify_grouped.py",         "$VD/render_loss/build/texsr_verify_grouped.py"),
    @("scripts/deeptex/render_gate.py",                  "$VD/render_loss/gates/render_gate.py"),
    @("scripts/matup/render_parity_torch.py",            "$VD/render_loss/gates/render_parity_torch.py"),
    @("scripts/deeptex/render_calibrate.py",             "$VD/render_loss/gates/render_calibrate.py"),
    @("scripts/compare/svbrdf_render_eval.py",           "$VD/render_loss/eval/svbrdf_render_eval.py"),
    @("scripts/compare/svbrdf_perceptual_eval.py",       "$VD/render_loss/eval/svbrdf_perceptual_eval.py"),
    @("scripts/deeptex/run_deeptex.ps1 -Stage eval-",    "$VD/render_loss/run.ps1 -Stage eval-"),
    @("``run_deeptex.ps1 -Stage eval-render``",          "``$VD/render_loss/run.ps1 -Stage eval-render``"),
    @("configs/deeptex/esrgan_matsynth_render_",         "$VD/render_loss/"),
    @("configs/deeptex/esrgan_matsynth",                 "$VD/matsynth/esrgan_matsynth"),
    @("configs/deeptex/esrgan_paper.yaml",               "$VD/paper/esrgan_paper.yaml"),
    @("configs/deeptex/psnr_texsr.yaml",                 "$VD/fixed/psnr_texsr.yaml"),
    @("configs/deeptex/esrgan_scaled.yaml",              "$VD/fixed/esrgan_scaled.yaml"),
    @("configs/deeptex/esrgan_asconfigured.yaml",        "$VD/fixed/esrgan_asconfigured.yaml")
)
$OLD_TRAIN = "/workspace/scripts/deeptex/deeptex_train.py"
$OLD_CMP   = "/workspace/scripts/deeptex/deeptex_compare.py"
$MS        = "/workspace/runs/deeptex/datasets/matsynth"
# The old ps1's $script:MatsynthRecipes (run_deeptex.ps1:539-543), for rule R.
$RECIPES = @{
    all       = [ordered]@{ baseColor = 1; normal = 1; roughness = 1; metallic = 1; height = 1 }
    chratio   = [ordered]@{ baseColor = 3; normal = 3; roughness = 1; metallic = 1; height = 1 }
    basecolor = [ordered]@{ baseColor = 1 }
}

function Convert-OldText([string]$s) {
    foreach ($p in $PATH_SUBS) { $s = $s.Replace($p[0], $p[1]) }
    return $s
}

function Find-Seq([string[]]$a, [string[]]$seq) {
    for ($i = 0; $i -le $a.Count - $seq.Count; $i++) {
        $ok = $true
        for ($j = 0; $j -lt $seq.Count; $j++) { if ($a[$i + $j] -cne $seq[$j]) { $ok = $false; break } }
        if ($ok) { return $i }
    }
    return -1
}

function Convert-OldCall([string[]]$argv0) {
    $a = New-Object "System.Collections.Generic.List[string]"
    foreach ($x in $argv0) { $a.Add((Convert-OldText $x)) }
    # E
    $i = $a.IndexOf($OLD_CMP)
    if ($i -ge 0) {
        $m = Find-Seq $a.ToArray() @("--maps", "all", "--scalar-collapse", "mean")
        if ($m -ge 0) { $a.RemoveRange($m, 4); $a[$i] = "/workspace/$VD/matsynth/eval_arms.py" }
        else { $a[$i] = "/workspace/$VD/infer/deeptex_compare.py" }
    }
    # T
    $i = $a.IndexOf($OLD_TRAIN)
    if ($i -ge 0) {
        $exp = if (@($a | Where-Object { $_ -like "*/$VD/render_loss/*" }).Count) { "render" } else { "fixed" }
        $a[$i] = "/workspace/$VD/paper_loop/main.py"
        $a.InsertRange($i + 1, [string[]]@("--experiment", $exp))
    }
    # R
    $i = $a.IndexOf("/workspace/$VD/upstream_runner/texsr_build_tfrecord.py")
    $o = $a.IndexOf("--out")
    if ($i -ge 0 -and $o -gt $i -and $a[$o + 1] -match '/matsynth_(\w+)_lrx4\.tfrecord$' -and $RECIPES.ContainsKey($Matches[1])) {
        $R = $Matches[1]
        $want = @()
        foreach ($m in $RECIPES[$R].Keys) {
            $want += @("--hr", "$MS/$m/HR", "--lr", "$MS/$m/LRx4")
            if ($R -ne "all") { $want += @("--repeat", "$($RECIPES[$R][$m])") }
        }
        $have = @($a.GetRange($i + 1, $o - $i - 1))
        if (($have -join "`n") -ceq ($want -join "`n")) {
            $a.RemoveRange($i, $o - $i)
            $a.InsertRange($i, [string[]]@("/workspace/$VD/matsynth/recipes.py", "--recipe", $R, "--root", $MS))
        }
    }
    return [string[]]$a.ToArray()
}

# ------------------------------------------------------------- fake docker

$global:L5 = @{ sandbox = ""; calls = $null; modes = @{}; ps = @() }

function Get-L5Arg([string[]]$argv, [string]$flag) {
    $i = [array]::IndexOf($argv, $flag)
    if ($i -ge 0 -and ($i + 1) -lt $argv.Count) { return $argv[$i + 1] }
    return $null
}

function ConvertTo-L5Path([string]$p) {
    $rel = ($p -replace '^/workspace/', '').TrimEnd('/') -replace '/', '\'
    return (Join-Path $global:L5.sandbox $rel)
}

function Get-L5Cfg([string]$hostCfg, [string]$key) {
    $m = Select-String -Path $hostCfg -CaseSensitive -Pattern "^${key}:\s*(?:!!\w+\s+)?['""]?([^'""#\s]*)['""]?" |
         Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value }
    return $null
}

function Touch([string]$path, [string]$text = "") {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null
    [IO.File]::WriteAllText($path, $text)
}

# ckpt-N in a host directory: .index/.data, its step in ckpt-N.step, and the
# `checkpoint` listing CopyListedCheckpoints reads. $old: back-date it, so an SR
# manifest written later counts as newer (Get-CompleteEval).
function New-L5Ckpt([string]$hostDir, [int]$n, [int]$step, [switch]$Old) {
    New-Item -ItemType Directory -Force -Path $hostDir | Out-Null
    foreach ($x in "data-00000-of-00001", "index", "step") {
        $f = Join-Path $hostDir "ckpt-$n.$x"
        [IO.File]::WriteAllText($f, $(if ($x -eq "step") { "$step" } else { "" }))
        if ($Old) { (Get-Item $f).LastWriteTimeUtc = [datetime]"2026-01-01" }
    }
    $names = @(Get-ChildItem $hostDir -Filter "ckpt-*.index" | ForEach-Object { $_.BaseName } |
               Sort-Object { [int]($_ -replace 'ckpt-', '') })
    $lines = @("model_checkpoint_path: ""$($names[-1])""") + @($names | ForEach-Object { "all_model_checkpoint_paths: ""$_""" })
    [IO.File]::WriteAllText((Join-Path $hostDir "checkpoint"), ($lines -join "`n") + "`n")
}

function Get-L5Latest([string]$hostDir) {
    $c = @(Get-ChildItem $hostDir -Filter "ckpt-*.index" -ErrorAction SilentlyContinue |
           Sort-Object { [int]($_.BaseName -replace 'ckpt-', '') })
    if (-not $c) { return $null }
    $n = [int]($c[-1].BaseName -replace 'ckpt-', '')
    return @{ n = $n; step = [int](Get-Content (Join-Path $hostDir "ckpt-$n.step")) }
}

function nvidia-smi {
    $a = "$args"
    if ($a -match "memory.used") { return "1000" }
    if ($a -match "memory.total") { return "12288" }
    return @()
}

function Invoke-L5Trainer([string[]]$cmd, [string]$wd, [string]$mode) {
    $cfgCtr = Get-L5Arg $cmd "--cfg_path"
    $cfgHost = if ($cfgCtr -like "/workspace/*") { ConvertTo-L5Path $cfgCtr } else { ConvertTo-L5Path "$wd/$cfgCtr" }
    $niter = [int](Get-L5Cfg $cfgHost "niter")
    $sub = Get-L5Cfg $cfgHost "sub_name"
    $max = Get-L5Arg $cmd "--max-steps"
    if ($max) { $niter = [int]$max }
    $root = Get-L5Arg $cmd "--checkpoint-root"
    if (-not $root) { $root = Get-L5Cfg $cfgHost "checkpoint_root" }
    if (-not $root) { $root = if ($wd) { "$wd/checkpoints" } else { "/workspace/runs/deeptex/checkpoints" } }
    $dir = ConvertTo-L5Path "$root/$sub"
    Write-Error "2026-10-31 I tensorflow/stream_executor/cuda/cuda_blas.cc: cuBLAS (fake stderr line)"
    switch ($mode) {
        "oom"  { "ResourceExhaustedError: OOM when allocating tensor"; $global:LASTEXITCODE = 1; return }
        "nan"  { "[fatal] non-finite loss at step 7"; $global:LASTEXITCODE = 4; return }
        "cpu"  { "Skipping registering GPU devices..."; $global:LASTEXITCODE = 0; return }
        "data" { "DataLossError: corrupted record at 12"; $global:LASTEXITCODE = 1; return }
    }
    $last = Get-L5Latest $dir
    $start = 0
    if ($last) { $start = $last.step; "[*] resumed from $root/$sub/ckpt-$($last.n) at step $start" }
    "[init] step=$start g_sha=0123456789abcdef d_sha=fedcba9876543210"
    $every = Get-L5Arg $cmd "--log-every"; $every = if ($every) { [int]$every } else { 100 }
    $mon = Get-L5Arg $cmd "--render-monitor-every"
    $w = Get-L5Arg $cmd "--w-render"
    for ($s = $start + $every; $s -le $niter; $s += $every) {
        "[metrics] step=$s loss_G=1.0000 steps_per_sec=2.500 gpu_mem_mb=5000"
    }
    # the last step is always logged (a run shorter than --log-every included)
    if (($niter - $start) % $every) { "[metrics] step=$niter loss_G=1.0000 steps_per_sec=2.500 gpu_mem_mb=5000" }
    if ($mon) {
        "[render] step=1 l21=1.0"
        "[render] step=$niter l21=$(if ($w -and $w -ne '0') { '0.5' } else { '0.9' })"
    }
    New-L5Ckpt $dir $(if ($last) { $last.n + 1 } else { 1 }) $niter
    "training done"
}

function Invoke-L5Sr([string[]]$cmd) {
    $out = ConvertTo-L5Path (Get-L5Arg $cmd "--out")
    $root = Get-L5Arg $cmd "--root"; if (-not $root) { $root = "/workspace/runs/matsynth" }
    $mat = Get-L5Arg $cmd "--material"
    $mats = if ($mat) { @($mat) } else {
        @(Get-ChildItem (ConvertTo-L5Path $root) -Directory | Where-Object { Test-Path (Join-Path $_.FullName "SVBRDF") } |
          ForEach-Object { $_.Name }) }
    $prefix = Get-L5Arg $cmd "--ckpt"
    $step = 0
    if ($prefix) { $step = [int](Get-Content ((ConvertTo-L5Path $prefix) + ".step")) }
    else {
        $d = Get-L5Arg $cmd "--ckpt-dir"
        if ($d) { $l = Get-L5Latest (ConvertTo-L5Path $d); if ($l) { $prefix = "$d/ckpt-$($l.n)"; $step = $l.step } }
    }
    foreach ($m in $mats) {
        foreach ($p in "baseColor", "normal", "roughness", "metallic", "height", "render") {
            Touch (Join-Path $out "$m\$p.png")
        }
        Touch (Join-Path $out "$m\sr_manifest.json") ((@{ step = $step; checkpoint = $prefix } | ConvertTo-Json -Compress))
    }
    "[sr] $($mats.Count) material(s) at step $step"
}

function docker {
    [string[]]$argv = @($args | ForEach-Object { "$_" })
    [void]$global:L5.calls.Add($argv)
    $global:LASTEXITCODE = 0

    if ($argv[0] -eq "context" -and $argv[1] -eq "show") { return "desktop-linux" }
    if ($argv[0] -eq "ps") { return @($global:L5.ps | Where-Object { $_ }) }
    if ($argv[0] -ne "compose") { return }

    $k = [array]::IndexOf($argv, "run") + 1
    $wd = ""
    while ($argv[$k] -like "-*" -or $argv[$k] -eq "") {
        if ($argv[$k] -eq "--name") { $k += 2 } elseif ($argv[$k] -eq "") { $k++ } elseif ($argv[$k] -eq "-w") { $wd = $argv[$k + 1]; $k += 2 } else { $k++ }
    }
    $cmd = @($argv[($k + 1)..($argv.Count - 1)])
    $mode = $global:L5.modes["trainer"]

    if ($cmd[1] -eq "-c") {
        $py = $cmd[2]
        if ($py -match "glob\.glob") {
            foreach ($d in @($cmd | Select-Object -Skip 3)) {
                $h = ConvertTo-L5Path $d
                $c = @(Get-ChildItem $h -Filter "ckpt-*.index" -ErrorAction SilentlyContinue | Sort-Object Name)
                if (-not $c) { "[ckpt-none] $d" }
                foreach ($i in $c) { "[ckpt] $(Get-Content (Join-Path $h "$($i.BaseName).step")) $d/$($i.BaseName)" }
            }
            return
        }
        if ($py -match "need (==|>=) (\d+)") { "[ckpt] checkpoints/x/ckpt-1 step=$($Matches[2]) (need $($Matches[1]) $($Matches[2]))"; return }
        "[fake] -c"
        return
    }
    $script = Split-Path -Leaf $cmd[1]
    switch ($script) {
        { $_ -in "deeptex_train.py", "main.py", "run_upstream_train.py" } { Invoke-L5Trainer $cmd $wd $mode; return }
        { $_ -in "deeptex_compare.py", "eval_arms.py" } { Invoke-L5Sr $cmd; return }
        "render_gate.py" {
            $o = ConvertTo-L5Path (Get-L5Arg $cmd "--out")
            foreach ($p in (Get-L5Arg $cmd "--part") -split ',') {
                if ($p -eq "export") { Touch (Join-Path $o "parity_tf.npz") }
                Touch (Join-Path $o "$p.json") '{"pass": true}'
            }
            return
        }
        "render_parity_torch.py" {
            $o = ConvertTo-L5Path (Get-L5Arg $cmd "--out")
            Touch (Join-Path $o "parity_lights.json") '{"pass": true}'; Touch (Join-Path $o "parity_torch.json") '{"pass": true}'
            return
        }
        "render_calibrate.py" { Touch (ConvertTo-L5Path (Get-L5Arg $cmd "--out")) '{"pass": true, "w_star": 10.0}'; "[calib] w_star=10.0"; return }
        "texsr_build_grouped_tfrecord.py" { Touch (ConvertTo-L5Path (Get-L5Arg $cmd "--out")); return }
        "texsr_verify_grouped.py" {
            Touch (Join-Path (Split-Path -Parent (ConvertTo-L5Path (Get-L5Arg $cmd "--tfrecord"))) "verify_report.json") '{"ok": true}'
            "[verify] OK"; return
        }
        "svbrdf_render_eval.py" {
            $o = ConvertTo-L5Path (Get-L5Arg $cmd "--out")
            foreach ($f in "render_mat_a.png", "overview_render_pred.png", "summary.md") { Touch (Join-Path $o $f) }
            return
        }
        "deeptex_arms_report.py" {
            $o = ConvertTo-L5Path (Get-L5Arg $cmd "--out")
            foreach ($f in "montages\mat_a.png", "overview_baseColor.png", "summary.md", "per_material.md", "metrics_long.csv", "report.json") {
                Touch (Join-Path $o $f)
            }
            return
        }
        "deeptex_drift.py" {
            $o = ConvertTo-L5Path (Get-L5Arg $cmd "--out")
            foreach ($f in "drift.json", "drift.md", "sheet.png") { Touch (Join-Path $o $f) }
            return
        }
        "svbrdf_perceptual_eval.py" {
            $o = ConvertTo-L5Path (Get-L5Arg $cmd "--out")
            foreach ($f in "perceptual_summary.md", "perceptual_report.json") { Touch (Join-Path $o $f) }
            return
        }
    }
    "[fake] ok"
}

# ------------------------------------------------------------- fixtures

function Join-SbPath([string]$sb, [string]$ctr) { return Join-Path $sb (($ctr -replace '^/workspace/', '') -replace '/', '\') }

function New-Base([string]$sb) {
    foreach ($m in "mat_a", "mat_b") { New-Item -ItemType Directory -Force -Path (Join-SbPath $sb "runs/matsynth/$m/SVBRDF") | Out-Null }
}

function New-Eval([string]$sb, [string]$ctrOut, [int]$step, [string]$prefix) {
    foreach ($m in "mat_a", "mat_b") {
        Touch (Join-Path (Join-SbPath $sb $ctrOut) "$m\sr_manifest.json") ((@{ step = $step; checkpoint = $prefix } | ConvertTo-Json -Compress))
    }
}

function New-Steps([string]$sb, [string]$ctrDir, [int[]]$steps) {
    $n = 0
    foreach ($s in $steps) { $n++; New-L5Ckpt (Join-SbPath $sb $ctrDir) $n $s -Old }
}

$CAL = '{"pass": true, "w_star": 10.0}'
$RG  = "/workspace/runs/deeptex/render_gate"
$RS  = "/workspace/runs/deeptex/_smoke_render"

function Need([bool]$cond, [string]$msg) { if (-not $cond) { return $msg } }

# params use OLD config paths; the new side gets them through rule P.
$SCENARIOS = @(
    # ---- paper
    @{ name = "prep"; arm = "paper"; params = @{ Stage = "prep" }; n = 4 }
    @{ name = "prep-force"; arm = "paper"; params = @{ Stage = "prep"; Force = $true }; n = 4 }
    @{ name = "smoke"; arm = "paper"; params = @{ Stage = "smoke" }; n = 2 }
    @{ name = "smoke-cpu-fallback"; arm = "paper"; params = @{ Stage = "smoke" }; modes = @{ trainer = "cpu" }; n = 1
       throws = "smoke train: cpu-fallback" }
    @{ name = "paper"; arm = "paper"; params = @{ Stage = "paper" }; n = 2; text = "paper training OK" }
    @{ name = "paper-maxsteps"; arm = "paper"; params = @{ Stage = "paper"; MaxSteps = 10 }; n = 2
       text = "scratch config /workspace/runs/deeptex/paper/esrgan_paper_maxsteps.yaml" }
    @{ name = "paper-keepall"; arm = "paper"; params = @{ Stage = "paper"; KeepAllCheckpoints = $true; MaxSteps = 200 }; n = 2
       text = "\[keep\] 1 checkpoint\(s\) in" }
    @{ name = "paper-resume"; arm = "paper"; params = @{ Stage = "paper"; Force = $true; MaxSteps = 200 }; n = 2
       setup = { param($sb) New-Steps $sb "/workspace/runs/deeptex/paper/checkpoints/esrgan_40k" @(100) }
       text = "resuming into existing checkpoints" }
    @{ name = "paper-delete"; arm = "paper"; params = @{ Stage = "paper"; Force = $true; DeleteCheckpoints = $true; MaxSteps = 200 }; n = 2
       setup = { param($sb) New-Steps $sb "/workspace/runs/deeptex/paper/checkpoints/esrgan_40k" @(100) }
       text = "moved existing checkpoints aside" }
    @{ name = "paper-oom"; arm = "paper"; params = @{ Stage = "paper" }; modes = @{ trainer = "oom" }; n = 1; throws = "out of memory" }
    @{ name = "paper-matsynth"; arm = "paper"; n = 2
       params = @{ Stage = "paper"; PaperConfig = "configs/deeptex/esrgan_matsynth.yaml"; PaperRoot = "runs/deeptex/matsynth_all" }
       text = "-Stage eval-arms -PaperConfig" }
    @{ name = "eval"; arm = "paper"; params = @{ Stage = "eval" }; n = 3
       setup = { param($sb) New-Steps $sb "/workspace/runs/deeptex/paper/checkpoints/esrgan_40k" @(40000) } }
    @{ name = "eval-ckptdir"; arm = "paper"; n = 3
       params = @{ Stage = "eval"; EvalTag = "fixed"; Config = "configs/deeptex/esrgan_scaled.yaml"; CkptDir = "/workspace/runs/deeptex/checkpoints/esrgan_texsr" } }
    @{ name = "all"; arm = "paper"; params = @{ Stage = "all"; MaxSteps = 200 }; n = 11 }
    @{ name = "preflight"; arm = "paper"; params = @{ Stage = "preflight" }; preflight = $true; n = 0; text = "vram +11288 MiB free" }
    @{ name = "preflight-stray"; arm = "paper"; params = @{ Stage = "preflight" }; preflight = $true; ps = @("texsr-deeptex-run-1")
       n = 0; throws = "containers already running: texsr-deeptex-run-1" }
    # ---- matsynth
    @{ name = "prep-matsynth"; arm = "matsynth"; params = @{ Stage = "prep-matsynth" }; n = 4; recipe = $true }
    @{ name = "prep-matsynth-chratio"; arm = "matsynth"; params = @{ Stage = "prep-matsynth"; Recipe = "chratio" }; n = 4; recipe = $true }
    @{ name = "prep-matsynth-basecolor"; arm = "matsynth"; params = @{ Stage = "prep-matsynth"; Recipe = "basecolor"; Force = $true }; n = 4; recipe = $true }
    @{ name = "eval-arms"; arm = "matsynth"; n = 3
       params = @{ Stage = "eval-arms"; PaperConfig = "configs/deeptex/esrgan_matsynth.yaml"; PaperRoot = "runs/deeptex/matsynth_all"; ExtraArms = @("matsynth_chratio") }
       setup = { param($sb) New-Item -ItemType Directory -Force -Path (Join-SbPath $sb "runs/deeptex/eval/matsynth_chratio") | Out-Null }
       text = "report -> runs\\deeptex\\report_matsynth_all" }
    @{ name = "eval-arms-tag"; arm = "matsynth"; n = 3
       params = @{ Stage = "eval-arms"; EvalTag = "mytag"; PaperConfig = "configs/deeptex/esrgan_matsynth_chratio.yaml"; PaperRoot = "runs/deeptex/matsynth_chratio" } }
    @{ name = "eval-arms-missing"; arm = "matsynth"; n = 0
       params = @{ Stage = "eval-arms"; ExtraArms = @("nope") }; throws = "-ExtraArms nope" }
    # ---- fixed
    @{ name = "pretrain"; arm = "fixed"; params = @{ Stage = "pretrain"; MaxSteps = 300 }; n = 1; text = "psnr training OK" }
    @{ name = "train"; arm = "fixed"; params = @{ Stage = "train"; MaxSteps = 300 }; n = 1; text = "esrgan training OK" }
    @{ name = "train-delete"; arm = "fixed"; n = 1
       params = @{ Stage = "train"; Config = "configs/deeptex/esrgan_scaled.yaml"; MaxSteps = 300; Force = $true; DeleteCheckpoints = $true }
       setup = { param($sb) New-Steps $sb "/workspace/runs/deeptex/checkpoints/esrgan_texsr" @(100) }
       text = "moved existing checkpoints aside" }
    @{ name = "train-resume"; arm = "fixed"; n = 1
       params = @{ Stage = "train"; Config = "configs/deeptex/esrgan_asconfigured.yaml"; MaxSteps = 300; Force = $true }
       setup = { param($sb) New-Steps $sb "/workspace/runs/deeptex/checkpoints/esrgan_asconfigured" @(100) }
       text = "resumed from" }
    @{ name = "train-nan"; arm = "fixed"; params = @{ Stage = "train" }; modes = @{ trainer = "nan" }; n = 1; throws = "non-finite loss" }
    @{ name = "train-data"; arm = "fixed"; params = @{ Stage = "pretrain" }; modes = @{ trainer = "data" }; n = 1; throws = "TFRecord problem" }
    # ---- render_loss
    @{ name = "prep-grouped-nopairs"; arm = "render_loss"; params = @{ Stage = "prep-grouped" }; n = 0; throws = "no MatSynth pairs" }
    @{ name = "prep-grouped"; arm = "render_loss"; params = @{ Stage = "prep-grouped" }; n = 2
       setup = { param($sb) New-Item -ItemType Directory -Force -Path (Join-SbPath $sb "$MS/_meta") | Out-Null }; text = "grouped record OK" }
    @{ name = "prep-grouped-exists"; arm = "render_loss"; params = @{ Stage = "prep-grouped" }; n = 1
       setup = { param($sb) Touch (Join-SbPath $sb "/workspace/runs/deeptex/datasets/matsynth_grouped/matsynth_grouped_lrx4.tfrecord") }
       text = "verifying it only" }
    @{ name = "gate-parity"; arm = "render_loss"; params = @{ Stage = "gate-render"; Gate = "parity" }; n = 3; text = "parity gate PASSED" }
    @{ name = "gate-calibrate"; arm = "render_loss"; params = @{ Stage = "gate-render"; Gate = "calibrate" }; n = 2
       setup = { param($sb)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_all/checkpoints_all/esrgan_40k" @(25000, 30000, 35000, 40000)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_chratio/checkpoints/esrgan_40k" @(30000, 35000, 40000) }
       text = "w_star = 10" }
    @{ name = "gate-calibrate-missing-step"; arm = "render_loss"; params = @{ Stage = "gate-render"; Gate = "calibrate" }; n = 1
       setup = { param($sb)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k" @(30000, 40000)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_chratio/checkpoints/esrgan_40k" @(30000, 35000, 40000) }
       throws = "no checkpoint holds step 35000" }
    @{ name = "gate-overfit"; arm = "render_loss"; params = @{ Stage = "gate-render"; Gate = "overfit" }; n = 2
       setup = { param($sb) Touch (Join-SbPath $sb "$RG/calibration.json") $CAL
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k" @(40000) }
       text = "overfit gate PASSED" }
    @{ name = "gate-overfit-nocal"; arm = "render_loss"; params = @{ Stage = "gate-render"; Gate = "overfit" }; n = 0; throws = "calibration.json - run -Stage gate-render" }
    @{ name = "gate-smoke"; arm = "render_loss"; params = @{ Stage = "gate-render"; Gate = "smoke" }; n = 7
       setup = { param($sb) Touch (Join-SbPath $sb "$RG/calibration.json") $CAL }
       text = "smoke gate PASSED" }
    @{ name = "gate-premise"; arm = "render_loss"; params = @{ Stage = "gate-render"; Gate = "premise" }; n = 7
       setup = { param($sb)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_all/checkpoints_all/esrgan_40k" @(30000, 35000, 40000)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_chratio/checkpoints/esrgan_40k" @(30000, 35000, 40000)
                 New-Eval $sb "/workspace/runs/deeptex/eval/matsynth_chratio_30k" 30000 "/workspace/runs/deeptex/matsynth_chratio/checkpoints/esrgan_40k/ckpt-1" }
       text = "\[skip\] /workspace/runs/deeptex/eval/matsynth_chratio_30k" }
    @{ name = "render-train-r1"; arm = "render_loss"; n = 2
       params = @{ Stage = "render-train"; Arm = "r1"; KeepAllCheckpoints = $true; MaxSteps = 200 }
       setup = { param($sb) Touch (Join-SbPath $sb "$RG/calibration.json") $CAL
                 Touch (Join-SbPath $sb "/workspace/runs/deeptex/datasets/matsynth_grouped/verify_report.json") }
       text = "render-r1 training OK" }
    @{ name = "render-train-r0b-resume"; arm = "render_loss"; n = 2
       params = @{ Stage = "render-train"; Arm = "r0b"; MaxSteps = 200 }
       setup = { param($sb) Touch (Join-SbPath $sb "/workspace/runs/deeptex/datasets/matsynth_grouped/verify_report.json")
                 New-Steps $sb "/workspace/runs/deeptex/render_r0b/checkpoints/esrgan_40k" @(100) }
       text = "this run RESUMED at step 100" }
    @{ name = "render-train-delete"; arm = "render_loss"; n = 2
       params = @{ Stage = "render-train"; Arm = "r0a"; MaxSteps = 200; Force = $true; DeleteCheckpoints = $true; KeepAllCheckpoints = $true }
       setup = { param($sb) Touch (Join-SbPath $sb "/workspace/runs/deeptex/datasets/matsynth_grouped/verify_report.json")
                 New-Steps $sb "/workspace/runs/deeptex/render_r0a/checkpoints/esrgan_40k" @(100)
                 New-Steps $sb "/workspace/runs/deeptex/render_r0a/checkpoints_all/esrgan_40k" @(100) }
       text = "moved existing checkpoints aside" }
    @{ name = "render-train-oom"; arm = "render_loss"; params = @{ Stage = "render-train"; Arm = "r0a" }; modes = @{ trainer = "oom" }; n = 1
       setup = { param($sb) Touch (Join-SbPath $sb "/workspace/runs/deeptex/datasets/matsynth_grouped/verify_report.json") }
       throws = "Set render.light_chunks: 4" }
    @{ name = "render-train-noprep"; arm = "render_loss"; params = @{ Stage = "render-train"; Arm = "r0a" }; n = 0; throws = "run -Stage prep-grouped first" }
    @{ name = "eval-render"; arm = "render_loss"; n = 11
       params = @{ Stage = "eval-render"; Arms = "render_r0a,render_r1"; Steps = "30000,40000" }
       setup = { param($sb)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k" @(40000)
                 New-Eval $sb "/workspace/runs/deeptex/eval/matsynth_all" 40000 "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-1"
                 New-Steps $sb "/workspace/runs/deeptex/render_r0a/checkpoints_all/esrgan_40k" @(30000, 35000, 40000)
                 New-Steps $sb "/workspace/runs/deeptex/render_r1/checkpoints/esrgan_40k" @(30000, 40000) }
       text = "montages -> " }
    @{ name = "eval-render-smoke"; arm = "render_loss"; n = 6
       params = @{ Stage = "eval-render"; Arms = "smoke_r1"; Steps = "200" }
       setup = { param($sb)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k" @(40000)
                 New-Eval $sb "/workspace/runs/deeptex/eval/matsynth_all" 40000 "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-1"
                 New-Steps $sb "$RS/smoke_r1/checkpoints/esrgan_40k" @(100, 200) } }
    @{ name = "eval-render-noref"; arm = "render_loss"; params = @{ Stage = "eval-render" }; n = 0; throws = "eval\matsynth_all is missing" }
    @{ name = "eval-perceptual"; arm = "render_loss"; n = 1
       params = @{ Stage = "eval-perceptual"; Arms = "render_r0a,render_r1"; Steps = "40000"; EvalDevice = "cpu" }
       setup = { param($sb)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k" @(40000)
                 New-Eval $sb "/workspace/runs/deeptex/eval/matsynth_all" 40000 "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-1"
                 foreach ($a in "r0a", "r1") {
                     New-Steps $sb "/workspace/runs/deeptex/render_$a/checkpoints/esrgan_40k" @(40000)
                     New-Eval $sb "/workspace/runs/deeptex/eval/render_${a}_40k" 40000 "/workspace/runs/deeptex/render_$a/checkpoints/esrgan_40k/ckpt-1"
                 }
                 Touch (Join-SbPath $sb "runs/montages/deeptex_render_loss/README.md") "# old readme`n" }
       text = "listed them in" }
    @{ name = "eval-perceptual-missing"; arm = "render_loss"; n = 0
       params = @{ Stage = "eval-perceptual"; Arms = "render_r0a"; Steps = "40000" }
       setup = { param($sb)
                 New-Steps $sb "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k" @(40000)
                 New-Eval $sb "/workspace/runs/deeptex/eval/matsynth_all" 40000 "/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-1" }
       throws = "run -Stage eval-render -Arms render_r0a -Steps 40000 first" }
)

# ------------------------------------------------------------- one side

function Format-Call([string[]]$argv, [string]$sb) {
    $parts = foreach ($a in $argv) {
        $a = [regex]::Replace($a, [regex]::Escape($sb), "<REPO>", "IgnoreCase")
        $a = $a -replace '\d{8}-\d{6}', '<STAMP>'
        if ($a -match '\s' -or $a -eq "") { '"' + ($a -replace "`n", '\n') + '"' } else { $a }
    }
    return "docker " + ($parts -join " ")
}

function Get-Tree([string]$root) {
    if (-not (Test-Path $root)) { return @() }
    $n = $root.Length + 1
    return @(Get-ChildItem $root -Recurse -Force | ForEach-Object {
        $_.FullName.Substring($n) + $(if ($_.PSIsContainer) { "\" } else { "" }) })
}

function Invoke-Side($sc, [string]$side) {
    $script = if ($side -eq "old") { $OLD_SCRIPT } else { "variants\deeptex\$($sc.arm)\run.ps1" }
    $extra  = if ($side -eq "old") { $OLD_EXTRA } else { $NEW_EXTRA }
    $sb  = Join-Path $outDir "sandbox\$($sc.name)\$side"
    foreach ($rel in @($script) + $extra) {
        $dst = Join-Path $sb $rel
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
        Copy-Item (Join-Path $repo $rel) $dst
        if ((Get-FileHash $dst).Hash -ne (Get-FileHash (Join-Path $repo $rel)).Hash) { throw "copy of $rel differs" }
    }
    New-Base $sb
    if ($sc.setup) { & $sc.setup $sb }

    $global:L5.sandbox = $sb
    $global:L5.calls   = New-Object System.Collections.ArrayList
    $global:L5.modes   = $(if ($sc.modes) { $sc.modes } else { @{} })
    $global:L5.ps      = $(if ($sc.ps) { $sc.ps } else { @() })

    $before = Get-Tree $sb
    $params = @{}
    foreach ($k in $sc.params.Keys) {
        $val = $sc.params[$k]
        if ($side -eq "new" -and $val -is [string]) { $val = Convert-OldText $val }
        $params[$k] = $val
    }
    if (-not $sc.preflight) { $params.SkipPreflight = $true }
    $lines = New-Object System.Collections.ArrayList
    $threw = ""; $where = ""
    try {
        & (Join-Path $sb $script) @params *>&1 | ForEach-Object { [void]$lines.Add("$_") }
    } catch {
        $threw = $_.Exception.Message
        $where = ($_.InvocationInfo.PositionMessage -split "`n")[0]
    }
    $after = Get-Tree $sb

    $isOld = ($side -eq "old")
    $norm = {
        param($s)
        $s = [regex]::Replace($s, [regex]::Escape($sb), "<REPO>", "IgnoreCase")
        $s = $s -replace '\d{8}-\d{6}', '<STAMP>'
        $s = $s -replace 'background job \d+', 'background job <ID>'
        if ($isOld) { $s = Convert-OldText $s }
        return $s
    }
    $tr = @($lines | ForEach-Object { & $norm $_ })
    # Rule R: the recipe line is recipes.py's now (the fake does not print it).
    if ($isOld) { $tr = @($tr | Where-Object { $_ -notmatch '^  recipe \w+ : ' }) }
    $raw = @($global:L5.calls)
    $calls = @($raw | ForEach-Object { Format-Call $(if ($isOld) { Convert-OldCall $_ } else { $_ }) $sb })
    return [pscustomobject]@{
        calls   = $calls
        compose = @($raw | Where-Object { $_[0] -eq "compose" }).Count
        transcript = $tr
        created = @($after | Where-Object { $before -notcontains $_ } | ForEach-Object { $_ -replace '\d{8}-\d{6}', '<STAMP>' } | Sort-Object)
        threw   = $(if ($threw) { & $norm $threw } else { "" })
        text    = (($calls + $tr) -join "`n")
        where   = $where
    }
}

function Compare-Lines([string]$what, [string[]]$a, [string[]]$b) {
    $out = New-Object System.Collections.ArrayList
    $n = [Math]::Max($a.Count, $b.Count)
    for ($i = 0; $i -lt $n; $i++) {
        $x = if ($i -lt $a.Count) { $a[$i] } else { "<missing>" }
        $y = if ($i -lt $b.Count) { $b[$i] } else { "<missing>" }
        if ($x -cne $y) { [void]$out.Add("${what}[$i]`n- $x`n+ $y") }
    }
    return $out
}

# ------------------------------------------------------------- run

$rows = New-Object System.Collections.ArrayList
$details = New-Object System.Collections.ArrayList

foreach ($sc in $SCENARIOS) {
    if ($Only.Count -and -not ($Only -contains $sc.name)) { continue }
    Write-Host ("{0,-30}" -f $sc.name) -NoNewline
    $r = @{}
    foreach ($side in "old", "new") { $r[$side] = Invoke-Side $sc $side }
    $diff = @()
    $diff += @(Compare-Lines "docker" $r.old.calls $r.new.calls)
    $diff += @(Compare-Lines "transcript" $r.old.transcript $r.new.transcript)
    $diff += @(Compare-Lines "created" $r.old.created $r.new.created)
    if ($r.old.threw -cne $r.new.threw) { $diff += "threw`n- $($r.old.threw)`n+ $($r.new.threw)" }
    $unmet = @()
    foreach ($side in "old", "new") {
        $x = $r[$side]
        $u = @(Need ($x.compose -eq $sc.n) "expected $($sc.n) container(s), got $($x.compose)")
        if ($sc.throws) { $u += Need ($x.threw -match [regex]::Escape($sc.throws)) "did not throw '$($sc.throws)' (threw: '$($x.threw)')" }
        else { $u += Need ($x.threw -eq "") "threw: $($x.threw)" }
        if ($sc.text) { $u += Need ($x.text -match $sc.text) "no /$($sc.text)/ in the transcript" }
        if ($sc.recipe -and $side -eq "new") {
            $u += Need (($x.calls -join "`n") -match "recipes\.py --recipe") "no recipes.py call"
        }
        $unmet += @($u | Where-Object { $_ } | ForEach-Object { "${side}: $_" })
    }
    $pass = ($diff.Count -eq 0 -and $unmet.Count -eq 0)
    Write-Host $(if ($pass) { "PASS" } else { "FAIL" }) -ForegroundColor $(if ($pass) { "Green" } else { "Red" })
    $note = "$($sc.arm): $($r.new.compose) container(s)" + $(if ($r.new.threw) { ", throws" } else { "" })
    [void]$rows.Add(@{ name = $sc.name; pass = $pass; note = $note })
    [void]$details.Add(@{ name = $sc.name; pass = $pass; r = $r; diff = $diff; unmet = $unmet })
}

# Every old -Stage value is covered by some arm.
$stages = @(([regex]::Match((Get-Content (Join-Path $repo $OLD_SCRIPT) -Raw), '\[ValidateSet\(("preflight"[^\]]+)\)\]').Groups[1].Value -split '[",\s]+') | Where-Object { $_ })
$covered = @($SCENARIOS | ForEach-Object { $_.params.Stage } | Sort-Object -Unique)
$missing = @($stages | Where-Object { $covered -notcontains $_ })
[void]$rows.Add(@{ name = "every-old-stage"; pass = ($missing.Count -eq 0)
                   note = $(if ($missing) { "not covered: $($missing -join ', ')" } else { "$($stages.Count) stages, all covered" }) })

$leak = @()
foreach ($root in (@($repo, $main) | Select-Object -Unique)) {
    $ld = Join-Path $root "runs\_logs"
    if (Test-Path $ld) {
        $leak += @(Get-ChildItem $ld -Directory | Where-Object { $_.CreationTime -ge $t0 } | ForEach-Object { $_.FullName })
    }
}
[void]$rows.Add(@{ name = "no-writes-outside"; pass = ($leak.Count -eq 0)
                   note = $(if ($leak) { "new: $($leak -join ', ')" } else { "no new runs\_logs entry in $repo or $main" }) })
$global:L5 = $null

# ------------------------------------------------------------- report

$allPass = -not ($rows | Where-Object { -not $_.pass })
$head = (git -C $repo rev-parse --short HEAD).Trim()
$L = New-Object System.Collections.ArrayList
[void]$L.Add("# L5 deeptex: run_deeptex.ps1 vs variants/deeptex/<arm>/run.ps1")
[void]$L.Add("")
[void]$L.Add("- stamp ``$stamp``, repo ``$repo`` at ``$head``")
[void]$L.Add("- old ``$OLD_SCRIPT`` sha256 ``$((Get-FileHash (Join-Path $repo $OLD_SCRIPT)).Hash.Substring(0,12))``")
foreach ($a in "paper", "matsynth", "fixed", "render_loss") {
    [void]$L.Add("- new ``variants/deeptex/$a/run.ps1`` sha256 ``$((Get-FileHash (Join-Path $repo "variants\deeptex\$a\run.ps1")).Hash.Substring(0,12))``")
}
[void]$L.Add("- sandboxes ``$outDir\sandbox``; docker intercepted by a PowerShell function (see the harness header)")
[void]$L.Add("- result: **$(if ($allPass) { 'ALL PASS' } else { 'FAIL' })**")
[void]$L.Add("")
[void]$L.Add("Mapping applied to the old side (rules T, E, R in the harness header), then this substring map (rule P):")
[void]$L.Add("")
[void]$L.Add("| old | new |")
[void]$L.Add("|---|---|")
foreach ($p in $PATH_SUBS) { [void]$L.Add("| ``$($p[0])`` | ``$($p[1])`` |") }
[void]$L.Add("")
[void]$L.Add("| scenario | result | note |")
[void]$L.Add("|---|---|---|")
foreach ($row in $rows) { [void]$L.Add("| $($row.name) | $(if ($row.pass) { 'PASS' } else { '**FAIL**' }) | $($row.note) |") }
foreach ($d in $details) {
    [void]$L.Add("")
    [void]$L.Add("## $($d.name): $(if ($d.pass) { 'PASS' } else { 'FAIL' })")
    if ($d.unmet) {
        [void]$L.Add(""); [void]$L.Add("Unmet expectations:")
        foreach ($u in $d.unmet) { [void]$L.Add("- $u") }
    }
    if ($d.diff) {
        [void]$L.Add(""); [void]$L.Add("Diff (- old after mapping, + new):"); [void]$L.Add(""); [void]$L.Add('```diff')
        foreach ($x in $d.diff) { [void]$L.Add($x) }
        [void]$L.Add('```')
    }
    [void]$L.Add(""); [void]$L.Add("docker calls (new):"); [void]$L.Add(""); [void]$L.Add('```')
    if ($d.r.new.calls.Count) { foreach ($c in $d.r.new.calls) { [void]$L.Add($c) } } else { [void]$L.Add("(none)") }
    [void]$L.Add('```')
    if ($d.r.new.threw) { [void]$L.Add(""); [void]$L.Add("throws: ``$($d.r.new.threw)`` ($($d.r.new.where))") }
}
$report = Join-Path $outDir "report.md"
$L | Out-File -FilePath $report -Encoding utf8
Write-Host "`nreport -> $report" -ForegroundColor Cyan
if ($allPass) { Write-Host "L5 deeptex: ALL PASS" -ForegroundColor Green; exit 0 }
Write-Host "L5 deeptex: FAIL" -ForegroundColor Red
exit 1
