$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CondaPrefix = if ($env:DCP_CONDA_PREFIX) { $env:DCP_CONDA_PREFIX } else { Join-Path $ProjectRoot '.conda-dcp-train' }
$CodeRoot = Join-Path $ProjectRoot 'code\Deep_Correlated_Prompting-main'
$ArrowRoot = if ($env:MMIMDB_DATA_ROOT) { $env:MMIMDB_DATA_ROOT } else { Join-Path $ProjectRoot 'data\mmimdb\arrow' }
$MissingTableRoot = if ($env:MISSING_TABLE_ROOT) { $env:MISSING_TABLE_ROOT } else { Join-Path $ProjectRoot 'data\mmimdb\missing_tables' }
$LocalPython = Join-Path $CondaPrefix 'python.exe'
$PythonExe = if ($env:PYTHON_BIN) {
    $env:PYTHON_BIN
} elseif (Test-Path -LiteralPath $LocalPython) {
    $LocalPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}

if (-not (Test-Path -LiteralPath (Join-Path $CodeRoot 'run.py'))) { throw "项目代码不存在: $CodeRoot" }
if (-not (Test-Path -LiteralPath $ArrowRoot)) { throw "Arrow 数据目录不存在: $ArrowRoot" }
if (-not (Test-Path -LiteralPath $MissingTableRoot)) { throw "缺失表目录不存在: $MissingTableRoot" }

if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $ProjectRoot 'cache\huggingface' }
if (-not $env:HUGGINGFACE_HUB_CACHE) { $env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub' }
if (-not $env:TRANSFORMERS_CACHE) { $env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers' }
if (-not $env:TORCH_HOME) { $env:TORCH_HOME = Join-Path $ProjectRoot 'cache\torch' }

Set-Location $CodeRoot

Write-Host "DCP reproduction environment ready." -ForegroundColor Green
Write-Host "Python: $PythonExe"
Write-Host "Project: $CodeRoot"
Write-Host "Arrow: $ArrowRoot"
Write-Host "Missing tables: $MissingTableRoot"
Write-Host "GPU: $((& $PythonExe -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"))"
Write-Host "Ready. No training command has been started."
