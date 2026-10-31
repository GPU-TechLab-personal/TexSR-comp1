<#
.SYNOPSIS
    Run every equivalence layer (variants/tests/README.md) and write one report.

.DESCRIPTION
    Runs, in order (each layer's own runner and report):

      L0     l0_static.py (every moves/*.json) + l0_skeleton.py       host, seconds
      L1     l1_fixtures.py (deeptex image) + l1_trainer.py            CPU, ~25 min
      L1gpu  l1_trainer.py --gpu                                       GPU, ~2 min
      L2     run_deeptex_eval.ps1 -Layers L2                           CPU, ~35 min
      L3     run_deeptex_eval.ps1 -Layers L3                           CPU + GPU, ~45 min
      L4     l4_gates.py                                               GPU, ~25 min
      L5     l5_deeptex.ps1 + l5_matup.ps1                             PowerShell, ~2 min
      L6     l6_experiments_all.sh (Git Bash; L0 + imports + CPU + GPU) ~21 min

    and collects the verdicts, minutes and the tail of every failing layer's
    output into

        runs\variants_verify\<stamp>\report.md

    The GPU layers run one after another and each refuses to start while
    another texsr container holds the GPU (variants/README.md rule 8). Every
    layer writes only under runs\variants_verify\ (L6 also under
    D:\experimentdata\texsr_exp\variants_verify\).

.PARAMETER Only
    Layer names to run (default: all), e.g. -Only L0,L5

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File variants\tests\run_all.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File variants\tests\run_all.ps1 -Only L0,L5,L6
#>
[CmdletBinding()]
param([string[]] $Only = @())

$ErrorActionPreference = "Stop"
$here  = $PSScriptRoot
$repo  = Split-Path -Parent (Split-Path -Parent $here)
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$out   = Join-Path $repo "runs\variants_verify\$stamp"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$Only  = @($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ })

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { throw "host python not found (L0, L1, L4 drivers run on the host)" }
$gitBash = "C:\Program Files\Git\bin\bash.exe"   # not WSL's bash.exe: L6 needs Git Bash paths
$ps = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File")
# Piped Python buffers its stdout until a step ends; unbuffered, each line shows as it is printed.
$env:PYTHONUNBUFFERED = "1"

# Each block runs its commands and leaves the worst exit code in $script:rc.
$layers = [ordered]@{
    "L0"    = { & $py (Join-Path $here "l0_static.py"); $script:rc = $LASTEXITCODE
                & $py (Join-Path $here "l0_skeleton.py"); $script:rc = [math]::Max($script:rc, $LASTEXITCODE) }
    "L1"    = { & docker run --rm -v "${repo}:/workspace" -w /workspace -e TF_CPP_MIN_LOG_LEVEL=2 -e PYTHONUNBUFFERED=1 `
                    texsr/deeptex:tf2.10.1 python variants/tests/l1_fixtures.py; $script:rc = $LASTEXITCODE
                if ($script:rc -eq 0) { & $py (Join-Path $here "l1_trainer.py"); $script:rc = $LASTEXITCODE } }
    "L1gpu" = { & $py (Join-Path $here "l1_trainer.py") --gpu; $script:rc = $LASTEXITCODE }
    "L2"    = { & powershell @ps (Join-Path $here "run_deeptex_eval.ps1") -Layers L2 -Stamp "$stamp-L2"; $script:rc = $LASTEXITCODE }
    "L3"    = { & powershell @ps (Join-Path $here "run_deeptex_eval.ps1") -Layers L3 -Stamp "$stamp-L3"; $script:rc = $LASTEXITCODE }
    "L4"    = { & $py (Join-Path $here "l4_gates.py"); $script:rc = $LASTEXITCODE }
    "L5"    = { & powershell @ps (Join-Path $here "l5_deeptex.ps1"); $script:rc = $LASTEXITCODE
                & powershell @ps (Join-Path $here "l5_matup.ps1"); $script:rc = [math]::Max($script:rc, $LASTEXITCODE) }
    "L6"    = { & $gitBash (Join-Path $here "l6_experiments_all.sh"); $script:rc = $LASTEXITCODE }
}

$rows = @()
foreach ($name in $layers.Keys) {
    if ($Only.Count -and -not ($Only -contains $name)) { continue }
    Write-Host "`n==== $name" -ForegroundColor Cyan
    $log = Join-Path $out "$name.log"
    $t0 = Get-Date
    $script:rc = -1
    # Native stderr (TF, docker) arrives as ErrorRecords in PowerShell 5.1.
    # Tee-Object would write UTF-16 there; the log is UTF-8 and flushed line by line.
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $w = New-Object System.IO.StreamWriter($log, $false, (New-Object System.Text.UTF8Encoding $false))
    $w.AutoFlush = $true
    try {
        & $layers[$name] 2>&1 | ForEach-Object { $line = "$_"; $w.WriteLine($line); $line } | Out-Host
    } catch {
        $line = "run_all: $name threw: $($_.Exception.Message)"; $w.WriteLine($line); $line | Out-Host
        $script:rc = 1
    } finally { $w.Close(); $ErrorActionPreference = $prev }
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    $rows += [pscustomobject]@{ Layer = $name; Verdict = $(if ($script:rc -eq 0) { "PASS" } else { "FAIL (exit $script:rc)" })
                                Minutes = $mins; Log = $log }
}

$head = (git -C $repo rev-parse --short HEAD).Trim()
$md = @("# variants equivalence - $stamp", "", "repo ``$repo`` at ``$head``", "",
        "| layer | verdict | minutes | log |", "|---|---|---|---|")
foreach ($r in $rows) { $md += "| $($r.Layer) | $($r.Verdict) | $($r.Minutes) | ``$($r.Log)`` |" }
foreach ($r in @($rows | Where-Object { $_.Verdict -ne "PASS" })) {
    $md += @("", "## $($r.Layer) - last 40 lines", "", '```')
    $md += @(Get-Content $r.Log -Tail 40 -Encoding UTF8)
    $md += '```'
}
[System.IO.File]::WriteAllText((Join-Path $out "report.md"), ($md -join "`n") + "`n",
                               (New-Object System.Text.UTF8Encoding $false))
$rows | Format-Table -AutoSize | Out-Host
Write-Host "report -> $(Join-Path $out 'report.md')"
if (@($rows | Where-Object { $_.Verdict -ne "PASS" }).Count) { exit 1 }
exit 0
