#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/road-damage-segmentation-portfolio
if [ -d ".venv" ]; then source .venv/bin/activate; fi
python - <<'CHECKPY'
import importlib.util, sys
missing=[m for m in ["torch","PIL","numpy","tqdm"] if importlib.util.find_spec(m) is None]
if missing:
    print("[ERROR] Missing base packages:", missing)
    sys.exit(1)
CHECKPY
python - <<'INSTALLPY'
import importlib.util, subprocess, sys
if importlib.util.find_spec("transformers") is None:
    print("[INSTALL] transformers safetensors")
    subprocess.check_call([sys.executable,"-m","pip","install","transformers","safetensors"])
else:
    print("[OK] transformers installed")
INSTALLPY
DATA_ROOT="datasets/pothole_binary_v6_aug/processed"
if [ ! -d "$DATA_ROOT/train/images" ]; then echo "[WARN] $DATA_ROOT not found. Falling back to datasets/pothole_binary_aug/processed"; DATA_ROOT="datasets/pothole_binary_aug/processed"; fi
if [ ! -d "$DATA_ROOT/train/images" ]; then echo "[ERROR] No augmented dataset found. Run V6 prepare first."; exit 1; fi
python scripts/verify_pairs.py "$DATA_ROOT"
python scripts/train_segformer_e3_baseline.py --data-root "$DATA_ROOT" --save-dir seg/runs/segformer_e3_v6_baseline --model-name nvidia/mit-b0 --epochs 60 --batch-size 2 --height 384 --width 512 --lr 0.00006 --weight-decay 0.01 --patience 12 --num-workers 4 --seed 42
python scripts/collect_segformer_e3_evidence.py --data-root "$DATA_ROOT" --run-dir seg/runs/segformer_e3_v6_baseline --checkpoint best.pth --out-zip LRS_SEGFORMER_E3_V6_EVIDENCE.zip
ls -lah LRS_SEGFORMER_E3_V6_EVIDENCE.zip
