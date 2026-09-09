[CmdletBinding()]
param(
    [switch]$WithOptional,
    [switch]$WithDev
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-Python {
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

$PythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to run the active Python interpreter."
}
if ($PythonVersion -ne "3.11") {
    throw "Python 3.11 is required; active interpreter is Python $PythonVersion."
}

Invoke-Python -m pip install --upgrade pip

# Keep the PyTorch CUDA wheel index scoped to PyTorch itself. Passing this
# index while installing all requirements can make normal PyPI packages
# unavailable or resolve them differently.
Invoke-Python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
Invoke-Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

if ($WithOptional) {
    Invoke-Python -m pip install -r (Join-Path $ProjectRoot "requirements-optional.txt")
}

if ($WithDev) {
    Invoke-Python -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt")
}

Invoke-Python -c "import torch, pytorch_lightning, numpy, chess, cupy, numba, bitsandbytes, tensorboard, matplotlib; print('imports ok'); print('torch', torch.__version__); print('torch CUDA', torch.version.cuda); print('CUDA available', torch.cuda.is_available()); print('Lightning', pytorch_lightning.__version__); print('NumPy', numpy.__version__)"
