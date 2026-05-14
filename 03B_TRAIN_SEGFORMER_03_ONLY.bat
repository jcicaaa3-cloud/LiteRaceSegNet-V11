@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [Train B] SegFormer-03 ONLY - Transformer baseline
echo ============================================================
echo This creates:
echo  seg\transformer_03\checkpoints\segformer_03_best.pth
echo It does NOT train LiteRaceSegNet.
echo.

if not exist "seg\transformer_03\hf_pretrained\segformer_b3_ade\config.json" (
  echo [WARN] HuggingFace SegFormer pretrained folder was not found.
  echo [INFO] Running setup first...
  call 02_SETUP_SEGFORMER_03_HF.bat
  if errorlevel 1 (
    echo [ERROR] SegFormer setup failed.
if not defined NO_PAUSE pause
    exit /b 1
  )
)

if not exist "datasets\pothole_binary\processed\train\images" (
  echo [ERROR] Dataset folder is missing.
  echo Expected:
  echo  datasets\pothole_binary\processed\train\images
  echo  datasets\pothole_binary\processed\train\masks
  echo  datasets\pothole_binary\processed\val\images
  echo  datasets\pothole_binary\processed\val\masks
if not defined NO_PAUSE pause
  exit /b 1
)

echo [CHECK] image-mask pairing with typo-tolerant matching...
python seg\tools\check_dataset_pairs.py --root datasets\pothole_binary\processed
if errorlevel 1 (
  echo [ERROR] Dataset pairing failed. Check seg\runs\dataset_pairing_reports\*.csv
if not defined NO_PAUSE pause
  exit /b 1
)

python seg\transformer_03\train_segformer_03.py --config seg\config\pothole_binary_segformer_03_train.yaml
if errorlevel 1 (
  echo [FAILED] SegFormer-03 training failed.
if not defined NO_PAUSE pause
  exit /b 1
)

echo.
echo [OK] SegFormer-03 training finished.
echo Checkpoint: seg\transformer_03\checkpoints\segformer_03_best.pth
if not defined NO_PAUSE pause