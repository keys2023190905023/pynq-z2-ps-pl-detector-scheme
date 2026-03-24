param(
    [string]$VivadoBat = "D:\vivado\2019.1\bin\vivado.bat"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$tcl = Join-Path $repoRoot "hardware\vivado_src\run_fullhw_detector_rtl_sim.tcl"

if (-not (Test-Path $VivadoBat)) {
    throw "Vivado batch executable not found: $VivadoBat"
}

& $VivadoBat -mode batch -source $tcl

if ($LASTEXITCODE -ne 0) {
    throw "Full-hardware detector RTL simulation failed with exit code $LASTEXITCODE"
}
