#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_CONFIG=${1:-seg/config/pothole_binary_literace_one_month_v3_balanced.yaml}
EPOCHS=${2:-}

if [ -n "$EPOCHS" ]; then
  python scripts/v11_make_ablation_configs.py --base "$BASE_CONFIG" --epochs "$EPOCHS"
else
  python scripts/v11_make_ablation_configs.py --base "$BASE_CONFIG"
fi

mkdir -p seg/runs/v11_ablation
for cfg in seg/config/v11_ablation/v11_*.yaml; do
  echo "============================================================"
  echo "[V11] Training ablation config: $cfg"
  echo "============================================================"
  python seg/train_literace.py --config "$cfg"
done

python seg/tools/v11_profile_models.py --config_dir seg/config/v11_ablation --out_csv seg/runs/v11_ablation/v11_model_profile.csv
python seg/tools/v11_summarize_ablation.py --runs_dir seg/runs/v11_ablation --profile_csv seg/runs/v11_ablation/v11_model_profile.csv --out_dir seg/runs/v11_ablation/_summary
