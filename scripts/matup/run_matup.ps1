<#
.SYNOPSIS
    Run MatUp over the extracted MatSynth materials, sequentially and resumably.

.DESCRIPTION
    Orchestrates `docker compose run` from the Windows host - it is not itself
    containerised. Everything it does by hand is documented in docs/RUNBOOK.md;
    this exists because doing it by hand across a dozen materials is where the
    mistakes happen.

    Three behaviours are not conveniences, they are corrections for failures this
    project has already hit:

    * Every run gets its OWN --render_dir / --upsamplings_dir / --output_folder.
      MatUp's defaults are shared paths and main_optim.py rmtree's them on
      success, so two concurrent runs delete each other's input. The victim dies
      in an unrelated-looking ZeroDivisionError (README operational note 6).

    * A run is judged by whether radiance_LR / radiance_Upscaled were REMOVED,
      not by the exit code. main_optim.py:103-104 deletes them only on success,
      which makes their absence the most reliable success signal available.

    * A GPU-loss signature aborts the whole batch immediately. When the GPU fell
      off the bus mid-batch, seven further materials failed back to back with
      nvml errors; continuing only wastes time and buries the first failure.

.EXAMPLE
    scripts\matup\run_matup.ps1
    scripts\matup\run_matup.ps1 -Material acg_metal_026
    scripts\matup\run_matup.ps1 -Force -PostProcess
#>
[CmdletBinding()]
param(
    [string]   $Material,
    [string]   $Root          = "runs/matsynth",
    [switch]   $Force,
    [switch]   $SkipPreflight,
    [switch]   $PostProcess,
    [int]      $TileWidth     = 17,
    [string[]] $AllowContainers = @()
)

$ErrorActionPreference = "Stop"
$repo    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$compose = Join-Path $repo "docker\compose.yaml"
$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir  = Join-Path $repo "runs\_logs\$stamp"

# MatSynth stores normals in the OpenGL convention ("We adopt the OpenGL standard
# for the Normal map"), so MatUp must be told, or it renders the optimisation
# against a Y-inverted surface.
$OGL = "--ogl_normals"

. (Join-Path (Split-Path -Parent $PSScriptRoot) "common\preflight.ps1")

function Say([string]$msg, [string]$colour = "Gray") { Write-Host $msg -ForegroundColor $colour }

# ---------------------------------------------------------------- preflight

function Invoke-Preflight {
    Say "preflight" "Cyan"

    # The docker-context and stray-container checks live in common/preflight.ps1
    # so this script and run_deeptex.ps1 cannot drift apart on the two-daemon trap,
    # which is the most-warned-about failure in this repo. The reason a stray
    # container matters differs per method, so it is passed in.
    Invoke-SharedPreflight -Repo $repo -AllowContainers $AllowContainers `
        -Reason ("a leftover MatUp container will delete this run's intermediates " +
                 "out from under it (README operational note 6).")

    $ck = Join-Path $repo "external\matup_egsr\pretrained_models\003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
    if (-not (Test-Path $ck)) { throw "SwinIR checkpoint missing: $ck" }
    $size = (Get-Item $ck).Length
    if ($size -ne 142473939) {
        throw "SwinIR checkpoint is $size bytes, expected 142473939. A partial " +
              "download makes sr_swinir.py exit(0) silently."
    }
    Say "  checkpoint     $size bytes" "DarkGray"

    # Costs one container start; cheaper than discovering it twelve minutes in.
    $cuda = docker compose -f $compose run --rm matup python -c "import torch; print(torch.cuda.is_available())" 2>&1 |
            Select-String -Pattern "^(True|False)$" | Select-Object -Last 1
    if ("$cuda".Trim() -ne "True") {
        throw "torch.cuda.is_available() is not True inside the container. MatUp " +
              "uses CUDA graphs and has no CPU fallback."
    }
    Say "  cuda           True" "DarkGray"
}

# ---------------------------------------------------------------- per material

function Test-AlreadyDone([string]$dir, [string]$name) {
    $hits = Get-ChildItem (Join-Path $dir "matup-output-$name") -Recurse -Filter "baseColor_opt.png" -ErrorAction SilentlyContinue
    return [bool]$hits
}

function Invoke-Material([string]$name, [string]$dir) {
    $w   = "/workspace/$Root/$name"
    $log = Join-Path $logDir "$name.log"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    # Windows PowerShell 5.1 wraps every stderr line from a native command in an
    # ErrorRecord, which $ErrorActionPreference='Stop' then throws on. MatUp
    # writes a torch.meshgrid UserWarning to stderr on every run, so relax the
    # preference around the call and flatten the records back to plain strings.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = docker compose -f $compose run --rm matup python main.py `
                 --upscaler swinir $OGL `
                 --tile_width $TileWidth `
                 --input_mat       "$w/SVBRDF/" `
                 --render_dir      "$w/radiance_LR/" `
                 --upsamplings_dir "$w/radiance_Upscaled/" `
                 --output_folder   "$w/matup-output-$name/" 2>&1 |
               ForEach-Object { $_.ToString() }
    } finally {
        $ErrorActionPreference = $prev
    }
    $sw.Stop()

    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $out | Out-File -FilePath $log -Encoding utf8

    $text = $out -join "`n"

    # Fatal signatures: the batch cannot recover from either, so say so.
    if ($text -match "nvml error|CUDA error: unknown error|failed to create shim task") {
        return @{ status = "gpu-lost"; seconds = $sw.Elapsed.TotalSeconds; log = $log }
    }
    if ($text -match "please download model at:") {
        return @{ status = "checkpoint"; seconds = $sw.Elapsed.TotalSeconds; log = $log }
    }

    # The real success test: MatUp removes its intermediates only when it finishes.
    $leftover = (Test-Path (Join-Path $dir "radiance_LR")) -or
                (Test-Path (Join-Path $dir "radiance_Upscaled"))

    # A failed run prints no loss line at all, and .Groups[1] on a null match
    # throws before the failure can be reported.
    $lastLoss = [regex]::Matches($text, "loss \d+ : ([\d.]+)") | Select-Object -Last 1
    $loss = if ($lastLoss) { $lastLoss.Groups[1].Value } else { "" }

    if ($leftover -or ($text -notmatch "\.\.\.done")) {
        return @{ status = "failed"; seconds = $sw.Elapsed.TotalSeconds; log = $log; loss = $loss }
    }
    return @{ status = "ok"; seconds = $sw.Elapsed.TotalSeconds; log = $log; loss = $loss }
}

# ---------------------------------------------------------------------- main

$rootPath = Join-Path $repo ($Root -replace "/", "\")
if (-not (Test-Path $rootPath)) { throw "material root not found: $rootPath" }

$names = if ($Material) { @($Material) }
         else { (Get-ChildItem $rootPath -Directory | Where-Object {
                    Test-Path (Join-Path $_.FullName "SVBRDF") }).Name }
if (-not $names) { throw "no materials with an SVBRDF/ folder under $rootPath" }

foreach ($n in $names) {
    if (-not (Test-Path (Join-Path (Join-Path $rootPath $n) "SVBRDF"))) {
        throw "no such material: $n (expected $rootPath\$n\SVBRDF)"
    }
}

if (-not $SkipPreflight) { Invoke-Preflight } else { Say "preflight skipped" "Yellow" }

Say "`n$($names.Count) material(s); logs -> runs\_logs\$stamp" "Cyan"

$results = @{}
$consecutiveFailures = 0
$i = 0

foreach ($name in $names) {
    $i++
    $dir = Join-Path $rootPath $name
    $tag = "[$i/$($names.Count)] $name"

    if (-not $Force -and (Test-AlreadyDone $dir $name)) {
        Say "$tag  skip (already has output)" "DarkGray"
        $results[$name] = @{ status = "skipped" }
        continue
    }

    Say "$tag  running..." "White"
    $r = Invoke-Material $name $dir
    $results[$name] = $r

    # `break` inside a switch exits only the switch in PowerShell, never the
    # enclosing foreach - so the abort decision is carried out here instead.
    $abort = $false
    switch ($r.status) {
        "ok" {
            Say ("  ok    {0,6:N0}s   final loss {1}" -f $r.seconds, $r.loss) "Green"
            $consecutiveFailures = 0
        }
        "gpu-lost" {
            Say "  GPU LOST - aborting the batch." "Red"
            Say "  The GPU has dropped off the bus. Nothing further will run until" "Red"
            Say "  the machine is rebooted (docs/OPEN-ITEMS.md #10). Log: $($r.log)" "Red"
            $abort = $true
        }
        "checkpoint" {
            Say "  checkpoint/CWD problem - aborting. Log: $($r.log)" "Red"
            $abort = $true
        }
        default {
            Say ("  FAILED {0,6:N0}s   log: {1}" -f $r.seconds, $r.log) "Red"
            $consecutiveFailures++
            if ($consecutiveFailures -ge 2) {
                Say "  two consecutive failures - aborting rather than burning the list." "Red"
                $abort = $true
            }
        }
    }
    if ($abort) { break }
}

# ------------------------------------------------------------------- summary

Say "`nsummary" "Cyan"
"{0,-30} {1,-10} {2,8} {3}" -f "material", "status", "sec", "final loss"
foreach ($name in $names) {
    $r = $results[$name]
    if (-not $r) { continue }
    "{0,-30} {1,-10} {2,8:N0} {3}" -f $name, $r.status,
        $(if ($r.seconds) { $r.seconds } else { 0 }),
        $(if ($r.loss) { $r.loss } else { "" })
}

# @() is load-bearing: without it a single surviving hashtable reports its own
# key count instead of 1.
$ok  = @($results.Values | Where-Object { $_.status -eq "ok" }).Count
$bad = @($results.Values | Where-Object { $_.status -notin @("ok", "skipped") }).Count
Say "`n$ok ok, $bad failed" $(if ($bad) { "Yellow" } else { "Green" })

function Invoke-Step([string]$label, [string]$service, [string[]]$cmd) {
    Say "`n-- $label" "Cyan"
    # Same stderr-as-ErrorRecord problem as the MatUp call: the matup image warns
    # about torch.meshgrid on every run.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        docker compose -f $compose run --rm $service @cmd 2>&1 |
            ForEach-Object { $_.ToString() } |
            Where-Object { $_ -notmatch "UserWarning|_VF\.meshgrid" }
    } finally {
        $ErrorActionPreference = $prev
    }
}

if ($PostProcess -and $bad -eq 0) {
    Say "`npost-processing chain" "Cyan"
    # Order matters. height and render produce the two columns MatUp does not
    # write, postprocess normalises everything into compare/, montage reads only
    # from there - so montage must come last and the other two must come first.
    Invoke-Step "height reconstruction (paper 3.3)" dataprep `
        @("python", "scripts/matup/matup_height.py", "--all")
    Invoke-Step "LR / bilinear / GT renders" matup `
        @("python", "/workspace/scripts/matup/matup_render.py", "--all")
    Invoke-Step "comparison sets + metrics" dataprep `
        @("python", "scripts/compare/matup_postprocess.py", "--all", "--metrics")
    Invoke-Step "montages" dataprep `
        @("python", "scripts/compare/matup_montage.py", "--all", "--collect", "/workspace/runs/montages")
} elseif ($PostProcess) {
    Say "`nskipping the post-processing chain: $bad material(s) failed" "Yellow"
}
