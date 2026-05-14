@echo off
setlocal
cd /d "%~dp0"
set BASE_CONFIG=seg/config/pothole_binary_literace_one_month_v3_balanced.yaml
if not "%~1"=="" set BASE_CONFIG=%~1
python scripts\v11_make_ablation_configs.py --base "%BASE_CONFIG%"
if errorlevel 1 exit /b 1

for %%F in (seg\config\v11_ablation\v11_*.yaml) do (
  echo ============================================================
  echo [V11] Training ablation config: %%F
  echo ============================================================
  python seg\train_literace.py --config "%%F"
  if errorlevel 1 exit /b 1
)

python seg\tools\v11_profile_models.py --config_dir seg\config\v11_ablation --out_csv seg\runs\v11_ablation\v11_model_profile.csv
python seg\tools\v11_summarize_ablation.py --runs_dir seg\runs\v11_ablation --profile_csv seg\runs\v11_ablation\v11_model_profile.csv --out_dir seg\runs\v11_ablation\_summary
endlocal
