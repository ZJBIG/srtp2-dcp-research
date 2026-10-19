$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = Join-Path $ProjectRoot 'code\Deep_Correlated_Prompting-main'

$CondaPrefix = if ($env:DCP_CONDA_PREFIX) {
    $env:DCP_CONDA_PREFIX
} else {
    Join-Path $ProjectRoot '.conda-dcp-train'
}
$LocalPython = Join-Path $CondaPrefix 'python.exe'
$PathPython = Get-Command python -ErrorAction SilentlyContinue
$PythonExe = if ($env:PYTHON_BIN) {
    $env:PYTHON_BIN
} elseif (Test-Path -LiteralPath $LocalPython) {
    $LocalPython
} elseif ($PathPython) {
    $PathPython.Source
} else {
    $null
}

$ArrowRoot = if ($env:MMIMDB_DATA_ROOT) {
    $env:MMIMDB_DATA_ROOT
} else {
    Join-Path $ProjectRoot 'data\mmimdb\arrow'
}
$MissingTableRoot = if ($env:MISSING_TABLE_ROOT) {
    $env:MISSING_TABLE_ROOT
} else {
    Join-Path $ProjectRoot 'data\mmimdb\missing_tables'
}
$ClipCacheRoot = if ($env:CLIP_CACHE_ROOT) {
    $env:CLIP_CACHE_ROOT
} else {
    Join-Path $ProjectRoot 'cache\clip'
}
$CheckpointPath = if ($env:ORIGINAL_DCP_PATH) {
    $env:ORIGINAL_DCP_PATH
} else {
    Join-Path $ProjectRoot 'experiments\dcp_mmimdb_reproduction\checkpoints\dcp_mmimdb_reproduction_seed0_seed0\version_2\checkpoints\epoch=3-step=507.ckpt'
}
$OutputRoot = if ($env:DCP_SMOKE_LOG_DIR) {
    $env:DCP_SMOKE_LOG_DIR
} else {
    Join-Path $ProjectRoot 'experiments\dcp_local_smoke'
}

if (-not $PythonExe) {
    throw 'No Python interpreter found. Set PYTHON_BIN, or create the local .conda-dcp-train environment.'
}

$RequiredPaths = [ordered]@{
    'code directory'          = $CodeRoot
    'run.py'                  = Join-Path $CodeRoot 'run.py'
    'MM-IMDb arrow directory' = $ArrowRoot
    'missing table directory' = $MissingTableRoot
    'CLIP cache directory'    = $ClipCacheRoot
    'Original DCP checkpoint' = $CheckpointPath
}
$MissingPaths = @(
    $RequiredPaths.GetEnumerator() |
        Where-Object { -not (Test-Path -LiteralPath $_.Value) } |
        ForEach-Object { '  - {0}: {1}' -f $_.Key, $_.Value }
)
if ($MissingPaths.Count -gt 0) {
    throw @"
Required inputs are missing:
$($MissingPaths -join "`n")

Override them with MMIMDB_DATA_ROOT, MISSING_TABLE_ROOT, CLIP_CACHE_ROOT and
ORIGINAL_DCP_PATH. The interpreter can be selected with PYTHON_BIN.
"@
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $ProjectRoot 'cache\huggingface' }
if (-not $env:HUGGINGFACE_HUB_CACHE) { $env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub' }
if (-not $env:TRANSFORMERS_CACHE) { $env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers' }
if (-not $env:TORCH_HOME) { $env:TORCH_HOME = Join-Path $ProjectRoot 'cache\torch' }
$env:PYTHONUNBUFFERED = '1'

Write-Host 'Local smoke test'
Write-Host ('Python: ' + $PythonExe)
Write-Host ('Code:   ' + $CodeRoot)
Write-Host ('Data:   ' + $ArrowRoot)
Write-Host ('Output: ' + $OutputRoot)

$CudaStatus = & $PythonExe -c 'import torch; print(torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) {
    throw ('Failed to import torch with ' + $PythonExe + ' (exit code ' + $LASTEXITCODE + ').')
}
Write-Host ('CUDA:   ' + $CudaStatus)

$RunArgs = @(
    'run.py',
    'with',
    'task_finetune_mmimdb',
    'reliability_learning',
    ('data_root=' + $ArrowRoot),
    ('missing_table_root=' + $MissingTableRoot),
    ('clip_cache_root=' + $ClipCacheRoot),
    ('original_dcp_path=' + $CheckpointPath),
    ('log_dir=' + $OutputRoot),
    'exp_name=local_smoke_relative_direct',
    'reliability_importance_mode=relative',
    'gate_supervision_mode=direct_task',
    'adapter_train_epochs=1',
    'reliability_predictor_lr=0.001',
    'reliability_adapter_lr=0.003',
    'reliability_gate_lr=0.001',
    'reliability_weight_decay=0.0001',
    'num_gpus=1',
    'num_nodes=1',
    'num_workers=0',
    'per_gpu_batchsize=2',
    'batch_size=64',
    'precision=16',
    'max_epoch=2',
    'max_steps=None',
    'fast_dev_run=False',
    'val_check_interval=1.0',
    'limit_train_batches=2',
    'limit_val_batches=2',
    'seed=0'
)

Push-Location $CodeRoot
try {
    & $PythonExe @RunArgs
    if ($LASTEXITCODE -ne 0) {
        throw ('Smoke test failed with exit code ' + $LASTEXITCODE)
    }
}
finally {
    Pop-Location
}

Write-Host 'Smoke test completed successfully.' -ForegroundColor Green
