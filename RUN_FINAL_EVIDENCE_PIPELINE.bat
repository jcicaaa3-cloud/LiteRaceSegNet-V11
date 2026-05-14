@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

echo ============================================================
echo LiteRaceSegNet final evidence pipeline
echo ============================================================
echo This pipeline does not require or generate a thesis/paper file.
echo It only prepares GitHub-visible evidence tables and reports.
echo.

echo [1/5] Readiness audit...
python seg	ools12_result_readiness_audit.py --repo . --out final_evidence\RESULT_READINESS_AUDIT_BEFORE.md

echo [2/5] Ablation config/profile candidates...
python scripts11_make_ablation_configs.py
python seg	ools11_profile_models.py --config_dir seg\config11_ablation --out_csv seguns11_ablation\model_profile.csv --device cpu --iters 10 --warmup 3

echo [3/5] CPU evidence...
call 08_CPU_LIGHTWEIGHT_EVIDENCE.bat

echo [4/5] GPU evidence...
call 09_GPU_ACCELERATION_EVIDENCE.bat

echo [5/5] Dual-device synthesis...
call 10_DUAL_DEVICE_RESEARCH_EVIDENCE.bat

python seg	ools12_result_readiness_audit.py --repo . --out final_evidence\RESULT_READINESS_AUDIT.md

echo.
echo [DONE] Check final_evidence\RESULT_READINESS_AUDIT.md and generated CSV/JSON/Markdown files.
pause
