# Continue-train ResUNet + SINet-V2 + Classifier on COD10K Train (archive 6 = v3).
# ESCNet is separate (WSL). Sequential to fit ~8 GB VRAM.
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -ErrorAction SilentlyContinue
$Proj = "C:\Users\batka\Downloads\TUKU DEEP LEARING\Camouflage_Breaker-main"
$Py = "C:\Users\batka\AppData\Local\Programs\Python\Python310\python.exe"
$LogDir = Join-Path $Proj "outputs\continue_archive6"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:CONTINUE_TRAIN = "1"
$env:CONTINUE_EPOCHS = "10"
$env:CONTINUE_LR = "5e-5"
$env:SINET_EPOCHS = "10"
$env:SINET_LR = "5e-5"
$env:SINET_BATCH_SIZE = "4"
$env:CLS_EPOCHS = "10"
$env:CLS_LR = "5e-5"

Set-Location $Proj

function Invoke-Step($Name, $Args) {
  $log = Join-Path $LogDir "$Name.log"
  Write-Host "==== START $Name ====" -ForegroundColor Cyan
  & $Py -u @Args 2>&1 | Tee-Object -FilePath $log
  if ($LASTEXITCODE -ne 0) {
    Write-Host "==== FAIL $Name exit=$LASTEXITCODE ====" -ForegroundColor Red
    exit $LASTEXITCODE
  }
  Write-Host "==== DONE $Name ====" -ForegroundColor Green
}

$stage = $env:CONTINUE_STAGE
if (-not $stage) { $stage = "all" }

if ($stage -eq "all" -or $stage -eq "resunet") {
  Invoke-Step "resunet" @("training\train_segmentation.py")
}
if ($stage -eq "all" -or $stage -eq "sinet") {
  Invoke-Step "sinetv2" @("training\train_sinetv2.py")
}
if ($stage -eq "all" -or $stage -eq "classifier") {
  Invoke-Step "classifier" @("training\train_classifier_final.py")
}

Write-Host "Continue-train pipeline finished for stage=$stage"
