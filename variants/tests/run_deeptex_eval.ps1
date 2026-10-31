<#
.SYNOPSIS
    DeepTextures data prep / inference / evaluation / reports: L0 + L2 + L3 equivalence.

.DESCRIPTION
    Runs, in order,
        L0  variants/tests/l0_static.py           (host, seconds)
        L2  variants/tests/l2_data.py             (docker, CPU)
        L3  variants/tests/l3_infer.py            (docker, CPU; GPU reference)
            variants/tests/l3_reports.py          (docker, CPU + GPU)
    and writes runs\variants_verify\deeptex_eval\<stamp>\report.md (per-layer PASS/FAIL,
    the failing diffs, and the committed files that could not be matched).
    Nothing outside runs\variants_verify\ is written: the checkout, runs\ and the
    submodules are mounted read-only into every container.

    Run it from the main checkout:
        powershell -ExecutionPolicy Bypass -File variants\tests\run_deeptex_eval.ps1
    or from a git worktree whose runs\ and external\ are empty:
        ... -Main C:\Users\admin\Desktop\Vsc_Workspace\TexSR-comp1

    GPU steps wait until no other texsr* container is running (variants/README.md rule 8).
    Times are in variants/tests/README_deeptex_eval.md.

.PARAMETER Layers
    Any of L0,L2,L3 (default all three).
.PARAMETER Main
    Checkout whose runs\ and external\ are mounted (default: this one).
.PARAMETER Data
    Host folder holding MatSynth\data\*.parquet, mounted at /data (L2 ms_pairs).
.PARAMETER Quick
    Smaller L2: MatSynth records from the first 300 materials per map, grouped record
    --limit 300. Says so in the report.
.PARAMETER NoGpu
    Skip every GPU step (the L3 render/perceptual sections, GPU drift and inference
    references). The remaining checks still run.
#>
param(
    [string]   $Stamp  = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [string[]] $Layers = @("L0", "L2", "L3"),
    [string]   $Main   = "",
    [string]   $Data   = "D:\experimentdata",
    [switch]   $Quick,
    [switch]   $NoGpu
)
$ErrorActionPreference = "Stop"
$tests = $PSScriptRoot
$repo  = (Resolve-Path (Join-Path $tests "..\..")).Path
if (-not $Main) { $Main = $repo }
$Layers = @($Layers | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$out = Join-Path $Main "runs\variants_verify\deeptex_eval\$Stamp"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$codes = [ordered]@{}
$t0 = Get-Date

function Step([string]$name, [string[]]$argv) {
    Write-Host "== $name" -ForegroundColor Cyan
    $s = Get-Date
    & python @argv
    $codes[$name] = $LASTEXITCODE
    Write-Host ("   {0} exit {1} ({2:N0} s)" -f $name, $LASTEXITCODE, ((Get-Date) - $s).TotalSeconds)
}

$common = @("--stamp", $Stamp, "--main", $Main)
if ($Layers -contains "L0") {
    Step "L0" @((Join-Path $tests "l0_static.py"), "--report", (Join-Path $out "l0.md"))
}
if ($Layers -contains "L2") {
    $a = @((Join-Path $tests "l2_data.py")) + $common
    if (Test-Path $Data) { $a += @("--data", $Data) } else { Write-Host "  ! $Data not found: ms_pairs will FAIL" -ForegroundColor Yellow }
    if ($Quick) { $a += @("--ms-subset", "300", "--grouped-limit", "300") }
    Step "L2" $a
}
if ($Layers -contains "L3") {
    $g = if ($NoGpu) { @("--no-gpu") } else { @() }
    Step "L3-infer" (@((Join-Path $tests "l3_infer.py")) + $common + $g)
    $only = if ($NoGpu) { @("--only", "montages,base,arms,drift,tiling") } else { @() }
    Step "L3-reports" (@((Join-Path $tests "l3_reports.py")) + $common + $g + $only)
}
& python (Join-Path $tests "l2_harness.py") report $out | Out-Null
$bad = @($codes.Keys | Where-Object { $codes[$_] -ne 0 })
Write-Host ("report -> {0}\report.md   ({1:N0} min)" -f $out, ((Get-Date) - $t0).TotalMinutes)
if ($bad) { Write-Host "FAIL: $($bad -join ', ')" -ForegroundColor Red; exit 1 }
Write-Host "PASS: $($codes.Keys -join ', ')" -ForegroundColor Green
exit 0
