# After ResUNet finishes, continue SINet-V2 then classifier on COD10K Train (archive 6).
$ErrorActionPreference = "Continue"
$Proj = "C:\Users\batka\Downloads\TUKU DEEP LEARING\Camouflage_Breaker-main"
$LogDir = Join-Path $Proj "outputs\continue_archive6"
$Py = "C:\Users\batka\AppData\Local\Programs\Python\Python310\python.exe"
$ResunetLog = Join-Path $LogDir "resunet.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Proj

Write-Host "Waiting for ResUNet continue-train to finish..."
while ($true) {
  if (Test-Path $ResunetLog) {
    $tail = Get-Content $ResunetLog -Tail 40 -ErrorAction SilentlyContinue | Out-String
    if ($tail -match "Training Complete|Best checkpoint|RESUNET_EXIT=0" -or (Select-String -Path $ResunetLog -Pattern "Training complete|Best model:" -Quiet -ErrorAction SilentlyContinue)) {
      # also require no python train_segmentation still running
      $still = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "train_segmentation" }
      if (-not $still) { break }
    }
  }
  Start-Sleep -Seconds 60
}
Write-Host "ResUNet finished. Starting SINet-V2..."

$env:CONTINUE_TRAIN = "1"
$env:SINET_EPOCHS = "10"
$env:SINET_LR = "5e-5"
$env:SINET_BATCH_SIZE = "4"
& $Py -u "training\train_sinetv2.py" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "sinetv2.log")
"SINET_EXIT=$LASTEXITCODE" | Tee-Object -FilePath (Join-Path $LogDir "sinetv2.log") -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "SINet finished. Starting classifier..."
$env:CONTINUE_TRAIN = "1"
$env:CLS_EPOCHS = "10"
$env:CLS_LR = "5e-5"
& $Py -u "training\train_classifier_final.py" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "classifier.log")
"CLS_EXIT=$LASTEXITCODE" | Tee-Object -FilePath (Join-Path $LogDir "classifier.log") -Append
Write-Host "Chain complete."
