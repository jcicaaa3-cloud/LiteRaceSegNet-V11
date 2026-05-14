"""Create a short markdown summary from current archived LiteRace evidence."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize current archived training/audit evidence")
    p.add_argument("--train_log", default="final_evidence/current_run_archive/literace_first_train/seg/runs/literace_boundary_degradation/train_log.csv")
    p.add_argument("--mask_audit", default="final_evidence/current_run_archive/current_force60_b4_mask_quality_audit.csv")
    p.add_argument("--out_md", default="final_evidence/current_run_archive/CURRENT_ARCHIVE_EVIDENCE_SUMMARY_V3.md")
    args = p.parse_args()

    train_rows = read_csv(Path(args.train_log))
    audit_rows = read_csv(Path(args.mask_audit))
    best = max(train_rows, key=lambda r: fnum(r, "miou_binary"), default={})
    flags = Counter(row.get("quality_flag", "UNKNOWN") for row in audit_rows)

    lines = [
        "# Current Archive Evidence Summary V3",
        "",
        "이 문서는 기존 evidence archive를 덮어쓰지 않고, 현재 상태를 한 달 연장 실험의 출발점으로 요약한다.",
        "",
        "## Training baseline",
        "",
    ]
    if best:
        lines += [
            f"- Best epoch by validation mIoU: {best.get('epoch', 'NA')}",
            f"- mIoU(binary): {fnum(best, 'miou_binary'):.4f}",
            f"- Damage IoU: {fnum(best, 'iou_damage'):.4f}",
            f"- Pixel accuracy: {fnum(best, 'pixel_acc'):.4f}",
        ]
    else:
        lines.append("- Training log not found.")
    lines += ["", "## Mask sanity audit", ""]
    if audit_rows:
        for flag, count in sorted(flags.items()):
            lines.append(f"- {flag}: {count}")
        lines += [
            "",
            "## Interpretation",
            "",
            "현재 archive는 모델이 손상 영역을 아예 못 찾는 상태라기보다는, 일부 샘플에서 도로 질감과 그림자를 손상으로 넓게 잡는 과검출 문제가 남아 있다.",
            "따라서 다음 한 달은 구조를 바꾸는 방향보다 threshold/min-area sweep, boundary/component 지표, 보수형 재학습 config로 증거를 보강하는 쪽이 안전하다.",
        ]
    else:
        lines.append("- Mask audit CSV not found.")

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
