<#
.SYNOPSIS
  Start the Inkling engine server with the accepted profile.

.DESCRIPTION
  Launches llama-server from the Inkling engine build with the flags and AMB_* switches of the
  fastest measured profile: Inkling-2-Mini UD-Q4_K_XL, DSpark block drafter with adaptive draft
  length (3, or 5 when acceptance is high), loop-aware expert cache, compute and KV lending,
  borrowed prefill upload ring, and the exact or KLD-checked fusions. 128k context by default.

  On five replayed real agent sessions at a 128k window this profile measured
  prefill ~2,330 tok/s (cold 43k prompt) and decode ~89 tok/s at 43k / ~81 tok/s at ~85k
  (61-104 by draft acceptance) on an RTX 5080 Laptop GPU (16 GB).

  Paths come from parameters or from environment variables:
    INK_BIN     folder that contains llama-server.exe
    INK_MODEL   Inkling-2-Mini-UD-Q4_K_XL.gguf
    INK_DSPARK  dspark-Inkling-2-Mini-Q8_0.gguf (the drafter; without it the no-draft profile is used)

  Context above ~160k does not fit a 16 GB card with this profile yet (open item, see README). Above 160000 layers and KV spill to RAM and decode drops to ~11-15 tok/s; the launcher warns.

.EXAMPLE
  .\start-inkling.ps1
.EXAMPLE
  .\start-inkling.ps1 -Context 90000 -Port 8081
.EXAMPLE
  .\start-inkling.ps1 -Model D:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf -DSpark D:\models\dspark-Inkling-2-Mini-Q8_0.gguf
#>
[CmdletBinding()]
param(
    [string]$Bin      = $env:INK_BIN,
    [string]$Model    = $env:INK_MODEL,
    [string]$DSpark   = $env:INK_DSPARK,
    [int]   $Context  = 128000,
    [int]   $Port     = 8080,
    [string]$BindHost = '127.0.0.1',
    [int]   $DraftThreads = 8,
    [switch]$NoDraft,
    [switch]$Stock,      # stock flags, no AMB_* switches (the measured floor), for comparison
    [switch]$DryRun,     # print the command and environment, do not start
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra     # anything else is passed to llama-server unchanged
)

$ErrorActionPreference = 'Stop'

if (-not $Bin) {
    foreach ($c in 'engine\build\bin') {
        $p = Join-Path $PSScriptRoot $c
        if (Test-Path (Join-Path $p 'llama-server.exe')) { $Bin = $p; break }
    }
}
if (-not $Bin -or -not (Test-Path (Join-Path $Bin 'llama-server.exe'))) {
    throw "llama-server.exe not found. Build the engine (docs/BUILD.md) and pass -Bin or set INK_BIN."
}

if (-not $Model -or -not (Test-Path $Model)) { throw "Model not found. Pass -Model or set INK_MODEL." }
if (-not $NoDraft -and (-not $DSpark -or -not (Test-Path $DSpark))) {
    Write-Warning "DSpark drafter not found; starting without a drafter (pass -DSpark or set INK_DSPARK)."
    $NoDraft = $true
}

$args_ = @(
    '--model', $Model, '--host', $BindHost, '--port', $Port,
    '--parallel', '1', '--flash-attn', 'on', '--no-context-shift',
    '-c', $Context, '--fit', 'on', '--fit-ctx', $Context, '--fit-target', '512',
    '--ubatch-size', '2048', '--moe-cache-mib', 'auto',
    '--load-mode', 'none', '--jinja', '--no-mmproj'
)
# passed through the environment: Windows PowerShell 5.1 strips the inner quotes of a JSON argument
$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = '{"reasoning_effort": "high"}'
if (-not $NoDraft) {
    $args_ += @('--model-draft', $DSpark, '--spec-type', 'draft-dspark',
                '--spec-draft-n-max', '5', '--spec-draft-n-ubatch', '256',
                '-td', $DraftThreads, '-tbd', $DraftThreads)
}
if ($Extra) { $args_ += $Extra }

# accepted switches (docs/SWITCHES.md); all are default-off in the engine
$envs = [ordered]@{}
if (-not $Stock) {
    $envs = [ordered]@{
        AMB_MOE_BIGBATCH_BYPASS = '1'; AMB_MOE_BATCH_COPY = '1'; AMB_IDS_ASYNC = '1'
        AMB_MOE_LOOP_DEDUP = '1'
        AMB_INK_GRAPH_LEAN = '1'; AMB_INK_FUSE_CLIP = '1'; AMB_INK_FUSE_CONV = '1'; AMB_INK_FUSE_CTRL = '1'
        AMB_INK_FUSE_SITU = '1'; AMB_INK_FUSE_MIX = '1'; AMB_INK_GQA_PACK = '1'
        AMB_INK_REL_MASK_DEC = '1'; AMB_MMVF_SMALL = '1'
        AMB_ELASTIC_COMPUTE = '1'; AMB_ELASTIC_KV = '3'; AMB_ELASTIC_COMPUTE_SMALL_MAX = '8'
        AMB_INK_PF_BIAS = '1'; AMB_INK_PF_CTRL = '1'; AMB_INK_PF_MIX = '1'
        AMB_PF_OVERLAP = '1'; AMB_PF_OVERLAP_SLOTS = '3'
        AMB_MOE_SCALE_DEV = '1'
    }
    if (-not $NoDraft) {
        $envs.AMB_DSPARK_TRIM = '1'; $envs.AMB_DSPARK_IDS = '1'; $envs.AMB_DSPARK_HEAD_WH = '1'; $envs.AMB_DSPARK_ADAPT = '1'
    }
    if ($Context -gt 160000) {
        Write-Warning "Context $Context does not fit a 16 GB card with this profile; layers and KV spill to RAM (slow). Making long windows fast is an open item."
    }
}

$exe = Join-Path $Bin 'llama-server.exe'
if ($DryRun) {
    $envs.GetEnumerator() | ForEach-Object { "set $($_.Key)=$($_.Value)" }
    "`"$exe`" " + (($args_ | ForEach-Object { if ("$_" -match '[\s"]') { '"' + ("$_" -replace '"', '\"') + '"' } else { "$_" } }) -join ' ')
    return
}
foreach ($k in $envs.Keys) { Set-Item -Path "Env:$k" -Value $envs[$k] }
Write-Host "Inkling engine: $exe"
Write-Host "Model: $Model  context: $Context  draft: $(if ($NoDraft) { 'none' } else { 'DSpark adaptive 3/5' })  port: $Port"
& $exe @args_
