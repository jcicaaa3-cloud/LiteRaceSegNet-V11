#!/usr/bin/env python3
"""Audit whether GitHub-visible evidence files are ready to replace TBD values.

This script does not fabricate missing numbers. It checks for the files that the
README/results dashboard expects and writes a Markdown readiness report.
"""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

EXPECTED = [
    ("Public reference metrics", "docs/reference_evidence/literace_reference_metrics.json", "reference_only"),
    ("V11 model params", "seg/runs/v11_ablation/v11_model_params_current.csv", "reference_or_final"),
    ("CPU comparison CSV", "final_evidence/02_metrics_and_compare_cpu/model_compare_summary.csv", "final_required"),
    ("GPU comparison CSV", "final_evidence/02_metrics_and_compare_gpu/model_compare_summary.csv", "final_required"),
    ("Dual-device final table", "final_evidence/06_report_ready/final_comparison_table.md", "final_required"),
    ("Boundary metrics CSV", "final_evidence/03_boundary_metrics/boundary_metrics.csv", "final_required_for_boundary_claim"),
    ("Failure taxonomy CSV", "final_evidence/04_failure_taxonomy/failure_taxonomy_manual.csv", "recommended"),
    ("Stress evaluation CSV", "final_evidence/05_stress_eval/stress_eval_summary.csv", "recommended_for_robustness_claim"),
]

TEMPLATES = [
    "evidence_templates/V12_FINAL_QUANTITATIVE_RESULTS_TEMPLATE.csv",
    "evidence_templates/V12_CPU_GPU_PROFILING_TEMPLATE.csv",
    "evidence_templates/V12_BOUNDARY_METRICS_TEMPLATE.csv",
    "evidence_templates/V12_STRESS_EVAL_TEMPLATE.csv",
    "evidence_templates/V12_FAILURE_TAXONOMY_TEMPLATE.csv",
    "evidence_templates/V12_CLAIM_GATE_TEMPLATE.csv",
]


def count_csv_rows(path: Path) -> int:
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            return max(0, sum(1 for _ in csv.DictReader(f)))
    except Exception:
        return -1


def read_reference(repo: Path):
    out = []
    p = repo / "docs/reference_evidence/literace_reference_metrics.json"
    if p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        out.append(
            "- README reference: "
            f"params_million={data.get('params_million')}, "
            f"fp32_mib={data.get('fp32_parameter_size_mib')}, "
            f"pixel_acc={data.get('pixel_accuracy')}, "
            f"binary_miou={data.get('binary_miou')}, "
            f"damage_iou={data.get('damage_iou')}"
        )
    p = repo / "seg/runs/v11_ablation/v11_model_params_current.csv"
    if p.exists():
        rows = list(csv.DictReader(open(p, encoding='utf-8-sig')))
        full = next((r for r in rows if r.get('variant') == 'v11_full_reference'), None)
        if full:
            out.append(f"- V11 full params: params_million={full.get('params_million')}, fp32_mib={full.get('fp32_mib')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.', help='Repository root')
    ap.add_argument('--out', default='final_evidence/V12_RESULT_READINESS_AUDIT.md')
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    lines = []
    lines.append('# V12 Result Readiness Audit')
    lines.append('')
    lines.append(f'- created_at: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f'- repo: `{repo}`')
    lines.append('')
    lines.append('## Expected evidence files')
    lines.append('')
    lines.append('| item | path | status | role | rows/size |')
    lines.append('|---|---|---:|---|---:|')
    missing_final = []
    for name, rel, role in EXPECTED:
        p = repo / rel
        status = 'OK' if p.exists() and p.stat().st_size > 0 else 'MISSING'
        if status == 'MISSING' and 'final_required' in role:
            missing_final.append(rel)
        if p.exists() and p.suffix.lower() == '.csv':
            sz = count_csv_rows(p)
        elif p.exists():
            sz = p.stat().st_size
        else:
            sz = 0
        lines.append(f'| {name} | `{rel}` | {status} | {role} | {sz} |')
    lines.append('')
    lines.append('## Evidence templates')
    lines.append('')
    lines.append('| template | status |')
    lines.append('|---|---:|')
    for rel in TEMPLATES:
        p = repo / rel
        status = 'OK' if p.exists() and p.stat().st_size > 0 else 'MISSING'
        lines.append(f'| `{rel}` | {status} |')
    lines.append('')
    lines.append('## Reference values found')
    lines.extend(read_reference(repo) or ['- No reference values found.'])
    lines.append('')
    lines.append('## Claim gate')
    if missing_final:
        lines.append('Final project claims must remain limited because the following required evidence files are missing:')
        for rel in missing_final:
            lines.append(f'- `{rel}`')
    else:
        lines.append('Required final comparison evidence files exist. Verify their contents before replacing TBD values.')
    lines.append('')
    lines.append('## Next action')
    lines.append('Run `10_DUAL_DEVICE_RESEARCH_EVIDENCE.bat` or `bash scripts/run_dual_device_evidence.sh`, then re-run this audit.')
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[OK] wrote {out}')

if __name__ == '__main__':
    main()
