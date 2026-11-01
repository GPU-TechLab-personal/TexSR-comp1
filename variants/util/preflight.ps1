<#
.SYNOPSIS
    Host-side checks shared by run_matup.ps1 and run_deeptex.ps1.

.DESCRIPTION
    Only the predicates that are genuinely common live here. Method-specific ones
    stay with their own orchestrator: the SwinIR byte-size check and
    torch.cuda.is_available() belong to MatUp, and the "does TF actually see the
    GPU" probe belongs to DeepTextures (deeptex_train.py runs it in-container,
    where it can also time a matmul).

    Factored out because the two-daemon trap is the single most-warned-about
    failure in this repo (README, docs/RUNBOOK.md 1, docs/OPEN-ITEMS.md #10) and
    two copies of that check would eventually disagree.

    Dot-source it:
        . "$PSScriptRoot\preflight.ps1"
        Invoke-SharedPreflight -Repo $repo -Reason "a stray container holds VRAM"
#>

function Test-DockerContext {
    $ctx = (docker context show).Trim()
    if ($ctx -ne "desktop-linux") {
        throw "docker context is '$ctx', expected 'desktop-linux'. A WSL context " +
              "builds into the native docker-ce, which shares no images, " +
              "containers or build cache with Docker Desktop (docs/OPEN-ITEMS.md #10)."
    }
    return $ctx
}

function Test-NoStrayContainers([string]$Reason) {
    $running = docker ps --format "{{.Names}}"
    if ($running) {
        throw "containers already running: $running. $Reason`n" +
              "  `docker compose run` survives Ctrl+C, so check `docker ps` and " +
              "stop them explicitly (docs/RUNBOOK.md 5)."
    }
}

function Invoke-SharedPreflight {
    param(
        [Parameter(Mandatory)][string] $Repo,
        [string] $Reason = "it may interfere with this run.",
        [string[]] $AllowContainers = @()
    )
    $ctx = Test-DockerContext
    Write-Host "  context        $ctx" -ForegroundColor DarkGray

    # `docker ps` against a dead daemon prints its error to stderr and returns
    # NOTHING on stdout - which is byte-identical to a clean machine. Without the
    # exit-code check below, a stopped Docker Desktop reads as "no stray
    # containers", preflight passes, and the first real step fails instead.
    # PowerShell 5.1 turns every stderr line from a native command into an
    # ErrorRecord, which $ErrorActionPreference='Stop' throws on BEFORE we ever
    # reach the exit-code test - so the daemon's own error message would surface
    # instead of ours. Same dance as run_matup.ps1:108-123.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $psOut = docker ps --format "{{.Names}}" 2>&1 | ForEach-Object { $_.ToString() }
        $psCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($psCode -ne 0) {
        throw "the Docker daemon is not reachable (``docker ps`` exited $psCode).`n" +
              "  Start Docker Desktop and wait for the whale icon to stop animating.`n" +
              "  Note ``docker context show`` still answers '$ctx' with the daemon down - " +
              "it reads client config, not the daemon."
    }
    $running = @($psOut | Where-Object { $_ })
    $blocking = @($running | Where-Object { $AllowContainers -notcontains $_ })
    if ($blocking) {
        throw "containers already running: $($blocking -join ', '). $Reason`n" +
              "  'docker compose run' survives Ctrl+C, so verify with 'docker ps' " +
              "and stop them explicitly (docs/RUNBOOK.md 5)."
    }
    if ($running) {
        Write-Host "  running (allowed) $($running -join ', ')" -ForegroundColor DarkGray
    } else {
        Write-Host "  no stray containers" -ForegroundColor DarkGray
    }
}
