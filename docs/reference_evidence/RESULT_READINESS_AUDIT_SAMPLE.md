# V12 Result Readiness Audit

- created_at: 2026-05-14T01:02:36
- repo: `/mnt/data/LiteRaceSegNet_GitHub_ProVisual_NoPaper`

## Expected evidence files

| item | path | status | role | rows/size |
|---|---|---:|---|---:|
| Public reference metrics | `docs/reference_evidence/literace_reference_metrics.json` | OK | reference_only | 438 |
| V11 model params | `seg/runs/v11_ablation/v11_model_params_current.csv` | OK | reference_or_final | 7 |
| CPU comparison CSV | `final_evidence/02_metrics_and_compare_cpu/model_compare_summary.csv` | MISSING | final_required | 0 |
| GPU comparison CSV | `final_evidence/02_metrics_and_compare_gpu/model_compare_summary.csv` | MISSING | final_required | 0 |
| Dual-device final table | `final_evidence/06_report_ready/final_comparison_table.md` | MISSING | final_required | 0 |
| Boundary metrics CSV | `final_evidence/03_boundary_metrics/boundary_metrics.csv` | MISSING | final_required_for_boundary_claim | 0 |
| Failure taxonomy CSV | `final_evidence/04_failure_taxonomy/failure_taxonomy_manual.csv` | MISSING | recommended | 0 |
| Stress evaluation CSV | `final_evidence/05_stress_eval/stress_eval_summary.csv` | MISSING | recommended_for_robustness_claim | 0 |

## Evidence templates

| template | status |
|---|---:|
| `evidence_templates/V12_FINAL_QUANTITATIVE_RESULTS_TEMPLATE.csv` | OK |
| `evidence_templates/V12_CPU_GPU_PROFILING_TEMPLATE.csv` | OK |
| `evidence_templates/V12_BOUNDARY_METRICS_TEMPLATE.csv` | OK |
| `evidence_templates/V12_STRESS_EVAL_TEMPLATE.csv` | OK |
| `evidence_templates/V12_FAILURE_TAXONOMY_TEMPLATE.csv` | OK |
| `evidence_templates/V12_CLAIM_GATE_TEMPLATE.csv` | OK |

## Reference values found
- README reference: params_million=0.1245, fp32_mib=0.475, pixel_acc=0.9157, binary_miou=0.7988, damage_iou=0.7029
- V11 full params: params_million=0.124509, fp32_mib=0.474964

## Claim gate
Final project claims must remain limited because the following required evidence files are missing:
- `final_evidence/02_metrics_and_compare_cpu/model_compare_summary.csv`
- `final_evidence/02_metrics_and_compare_gpu/model_compare_summary.csv`
- `final_evidence/06_report_ready/final_comparison_table.md`
- `final_evidence/03_boundary_metrics/boundary_metrics.csv`

## Next action
Run `10_DUAL_DEVICE_RESEARCH_EVIDENCE.bat` or `bash scripts/run_dual_device_evidence.sh`, then re-run this audit.
