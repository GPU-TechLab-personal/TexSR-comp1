<#
.SYNOPSIS
    The helpers every DeepTextures run.ps1 shares, moved out of
    scripts/deeptex/run_deeptex.ps1.

.DESCRIPTION
    Dot-source it AFTER the run.ps1's param block and its path variables
    ($repo, $compose, $expCompose, $stamp, $logDir, $EVAL_ROOT, $RGATE, $RSMOKE,
    $MS_PAIRS, $GROUPED, ...), which these functions read from script scope just
    as they did inside run_deeptex.ps1:

        . (Join-Path $repo "variants\util\lib.ps1")

    One copy of each helper. In particular the checkpoint keeper: Stage-Paper
    inlined the same Start-Job loop that Start-CheckpointKeeper /
    Stop-CheckpointKeeper wrap (old :428-457 and :947-969); paper/run.ps1 now calls
    the functions.

    Each function below is the old one with only the script and config paths
    changed (Get-ArmInfo: configs/deeptex/* -> variants/deeptex/<arm>/*;
    Invoke-CompareAtSteps: `deeptex_compare.py --maps all --scalar-collapse mean`
    -> matsynth/eval_arms.py, whose defaults those are).
    Arm tags are the registry's (variants/arms.json): Get-ArmInfo, Get-DisplayName.
#>

. (Join-Path $PSScriptRoot "preflight.ps1")

# Moved from (old line ranges):
#   CopyListedCheckpoints      run_deeptex.ps1:237-257
#   Say                        run_deeptex.ps1:165-165
#   Invoke-DeeptexPreflight    run_deeptex.ps1:169-202
#   Invoke-Step                run_deeptex.ps1:206-235
#   Get-FailureKind            run_deeptex.ps1:259-275
#   Assert-TrainingReached     run_deeptex.ps1:277-287
#   Clear-Checkpoints          run_deeptex.ps1:338-349
#   Get-HostPath               run_deeptex.ps1:639-643
#   Read-CfgValue              run_deeptex.ps1:645-652
#   Split-List                 run_deeptex.ps1:654-656
#   Get-StepTag                run_deeptex.ps1:658-661
#   ConvertTo-Double           run_deeptex.ps1:663-669
#   Format-Float               run_deeptex.ps1:671-673
#   Get-Median                 run_deeptex.ps1:675-681
#   Write-JsonFile             run_deeptex.ps1:683-689
#   Get-ArmInfo                run_deeptex.ps1:691-731 (now read from variants/arms.json)
#   Get-CheckpointSteps        run_deeptex.ps1:733-778
#   Resolve-StepPrefix         run_deeptex.ps1:780-796
#   Assert-CheckpointStep      run_deeptex.ps1:798-816
#   Get-CompleteEval           run_deeptex.ps1:818-852
#   Invoke-CompareAtSteps      run_deeptex.ps1:854-895
#   Get-RenderFailureKind      run_deeptex.ps1:897-906
#   Clear-SmokeDir             run_deeptex.ps1:908-917
#   Read-WStar                 run_deeptex.ps1:919-934
#   Show-GateResults           run_deeptex.ps1:936-944
#   Start-CheckpointKeeper     run_deeptex.ps1:946-959
#   Stop-CheckpointKeeper      run_deeptex.ps1:961-969
#   Assert-RenderVram          run_deeptex.ps1:971-1006

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

function Say([string]$msg, [string]$colour = "Gray") { Write-Host $msg -ForegroundColor $colour }

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

function Invoke-Step {
    # $ComposeFile: another compose file for $Service (eval-perceptual's exp
    # service); empty = docker\compose.yaml, as every other stage uses.
    # $NamePrefix: the container name prefix, default $script:StepNamePrefix or
    # "texsr-deeptex" (a run.ps1 of another method sets $script:StepNamePrefix).
    # It must start with "texsr-": the GPU-busy checks look for texsr* containers.
    param([string]$Label, [string]$Service, [string[]]$Cmd, [string]$LogName,
          [string]$WorkDir = "", [string]$ComposeFile = "", [string]$NamePrefix = "")

    Say "`n-- $Label" "Cyan"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir "$LogName.log"

    # Windows PowerShell 5.1 wraps every stderr line from a native command in an
    # ErrorRecord, which $ErrorActionPreference='Stop' then throws on. TF writes
    # the cuBLAS/NUMA lines (harmless, docs/OPEN-ITEMS.md #5) to stderr on EVERY
    # invocation, so this is mandatory here rather than merely prudent.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $pfx = if ($NamePrefix) { $NamePrefix } elseif ($script:StepNamePrefix) { $script:StepNamePrefix } else { "texsr-deeptex" }
    if ($pfx -notlike "texsr-*") { throw "Invoke-Step: container name prefix '$pfx' must start with 'texsr-'" }
    $name = "$pfx-$LogName-$stamp"
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

# The arm registry, variants/arms.json (also read by variants/util/arms.py).
function Get-ArmRegistry {
    if (-not $script:ArmRegistry) {
        $p = Join-Path $repo "variants/arms.json"
        $script:ArmRegistry = [System.IO.File]::ReadAllText($p) | ConvertFrom-Json
    }
    return $script:ArmRegistry
}

# Arm tag -> config, checkpoint directories, sub_name, from the registry. The tag
# is what names runs/deeptex/eval/<tag>_<k>k and the report columns. checkpoint_root
# "from_config" = the config's checkpoint_root key (where the trainer wrote; the
# render arms), else runs/deeptex/<tag>/checkpoints; {RSMOKE} = this run.ps1's
# smoke root (the gate-render -Gate smoke checkpoints, for a dry run of eval-render).
function Get-ArmInfo([string]$tag) {
    $reg = Get-ArmRegistry
    $e = $reg.arms.$tag
    if (-not $e) {
        $known = ($reg.arms.PSObject.Properties | ForEach-Object { $_.Name }) -join ", "
        throw "unknown arm tag '$tag'. Known (variants/arms.json): $known."
    }
    $cfg = $e.config
    if (-not (Test-Path (Join-Path $repo $cfg))) { throw "$tag : config $cfg does not exist" }
    $ckptRoot = $e.checkpoint_root
    if ($ckptRoot -eq "from_config") {
        $ckptRoot = Read-CfgValue $cfg "checkpoint_root"
        if (-not $ckptRoot) { $ckptRoot = "/workspace/runs/deeptex/$tag/checkpoints" }
    }
    $ckptRoot = $ckptRoot.Replace("{RSMOKE}", "$RSMOKE")
    if ($ckptRoot -notlike "/workspace/*") {
        throw "$tag : checkpoint_root '$ckptRoot' in $cfg is not under /workspace"
    }
    $sn = Read-CfgValue $cfg "sub_name"
    if (-not $sn) { throw "$tag : no sub_name in $cfg" }
    return [pscustomobject]@{
        Tag = $tag; Cfg = $cfg; SubName = $sn
        Method = $e.method
        Ckpt = "$ckptRoot/$sn"
        # Where -KeepAllCheckpoints copies to: checkpoints -> checkpoints_all, as
        # Stage-Paper names it.
        CkptAll = "${ckptRoot}_all/$sn"
    }
}

# Row name of an arm tag (the registry display, + " <NN>k" for a step-suffixed
# tag); an unknown tag shows as itself - as variants/util/arms.display_name.
function Get-DisplayName([string]$label) {
    $arms = (Get-ArmRegistry).arms
    if ($arms.$label) { return $arms.$label.display }
    if ($label -match '^(.+)_(\d+k)$' -and $arms.($Matches[1])) {
        return "$($arms.($Matches[1]).display) $($Matches[2])"
    }
    return $label
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
        # gone, scored something else - not reused. The checkpoint's file is the
        # manifest's checkpoint_file (a torch .pth, variants/util/arm_output.py),
        # else <checkpoint>.index (a TF prefix).
        $src = if ($j.checkpoint_file) { [string]$j.checkpoint_file } else { ([string]$j.checkpoint) + ".index" }
        $idx = Get-HostPath $src
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

# matsynth/eval_arms.py for every (arm, step) -> <evalBase>/<tag>_<k>k,
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
            @("python","/workspace/variants/deeptex/matsynth/eval_arms.py","--cfg","/workspace/$($p.Info.Cfg)",
              "--ckpt",$prefix,"--root",$EVAL_ROOT,"--out",$p.Out) "$logTag-sr-$($p.Label)"
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

# Stage-Paper (paper/run.ps1) calls these too; it used to inline the same Start-Job block.
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
