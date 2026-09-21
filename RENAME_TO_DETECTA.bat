@echo off
cd /d "C:\Users\batka\Downloads\TUKU DEEP LEARING"
if exist "DETECTA" (
  echo DETECTA already exists.
  exit /b 1
)
if not exist "Camouflage_Breaker-main" (
  echo Source Camouflage_Breaker-main not found.
  exit /b 1
)
ren "Camouflage_Breaker-main" DETECTA
if exist DETECTA (
  echo SUCCESS: renamed to DETECTA
) else (
  echo FAILED: close Cursor/terminals locking the folder, then re-run.
  exit /b 1
)
