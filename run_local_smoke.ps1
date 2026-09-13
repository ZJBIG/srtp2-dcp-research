$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeRoot = Join-Path $ProjectRoot 'code\Deep_Correlated_Prompting-main'
$PythonExe = Join-Path $ProjectRoot '.conda-dcp-train\python.exe'
$ArrowRoot = Join-Path $ProjectRoot 'data\mmimdb\arrow'
$MissingTableRoot = Join-Path $ProjectRoot 'data\mmimdb\missing_tables'
$ClipCacheRoot = Join-Path $ProjectRoot 'cache\clip'
$CheckpointPath = Join-Path $ProjectRoot 'experiments\dcp_mmimdb_reproduction\checkpoints\dcp_mmimdb_reproduction_seed0_seed0\version_2\checkpoints\epoch=3-step=507.ckpt'
$OutputRoot = Join-Path $ProjectRoot 'experiments\dcp_local_smoke'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw 'Local Python not found.'
}
foreach ($Path in @($CodeRoot, $ArrowRoot, $MissingTableRoot, $ClipCacheRoot, $CheckpointPath)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw 'Required path not found.'
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$env:HF_HOME = Join-Path $ProjectRoot 'cache\huggingface'
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers'
$env:TORCH_HOME = Join-Path $ProjectRoot 'cache\torch'
$env:PYTHONUNBUFFERED = '1'

Set-Location $CodeRoot

Write-Host 'Local smoke test'
Write-Host ('Python: ' + $PythonExe)
Write-Host ('Code:   ' + $CodeRoot)
Write-Host ('Data:   ' + $ArrowRoot)
Write-Host ('Output: ' + $OutputRoot)
Write-Host ('CUDA:   ' + (& $PythonExe -c 'import torch; print(torch.cuda.is_available())'))

& $PythonExe run.py with `
  task_finetune_mmimdb `
  reliability_learning `
  ('data_root=' + $ArrowRoot) `
  ('missing_table_root=' + $MissingTableRoot) `
  ('clip_cache_root=' + $ClipCacheRoot) `
  ('original_dcp_path=' + $CheckpointPath) `
  ('log_dir=' + $OutputRoot) `
  'exp_name=local_smoke_relative_direct' `
  reliability_importance_mode=relative `
  gate_supervision_mode=direct_task `
  adapter_train_epochs=1 `
  reliability_predictor_lr=0.001 `
  reliability_adapter_lr=0.003 `
  reliability_gate_lr=0.001 `
  reliability_weight_decay=0.0001 `
  num_gpus=1 `
  num_nodes=1 `
  num_workers=0 `
  per_gpu_batchsize=2 `
  batch_size=64 `
  precision=16 `
  max_epoch=2 `
  max_steps=None `
  fast_dev_run=False `
  val_check_interval=1.0 `
  limit_train_batches=2 `
  limit_val_batches=2 `
  seed=0

if ($LASTEXITCODE -ne 0) {
    throw ('Smoke test failed with exit code ' + $LASTEXITCODE)
}

Write-Host 'Smoke test completed successfully.' -ForegroundColor Green
