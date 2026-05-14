# LiteRaceSegNet project overview

LiteRaceSegNet은 도로 손상 semantic segmentation을 위한 경량 CNN 프로젝트입니다. 이 GitHub 공개용 패키지는 논문 원고가 아니라 **구현 코드, 구조도, 정성 결과 이미지, 실험 프로토콜, evidence template**을 보여주는 저장소입니다.

## 한 줄 설명

> Boundary degradation에 민감한 도로 손상 segmentation을 위해 detail branch, context branch, boundary-guided fusion을 결합한 lightweight CNN 프로젝트.

## 왜 필요한가

도로 손상은 작은 면적, 낮은 대비, 불규칙 경계, 얇은 crack 구조, 높은 background 비율을 동시에 가집니다. 따라서 전체 mIoU만으로 실제 mask 품질을 설명하기 어렵고, 특히 boundary erosion, thin-structure disappearance, false boundary activation 같은 오류가 중요합니다.

## 핵심 구성

| 구성 | 역할 |
|---|---|
| Detail branch | H/2 feature를 유지하여 작은 손상 경계와 얇은 구조 보존 |
| Context branch + LiteASPP | H/8 feature에서 도로 표면과 배경 문맥을 저비용으로 반영 |
| Boundary prediction head | GT mask에서 생성한 boundary target으로 auxiliary supervision |
| Boundary gate | boundary signal을 이용해 detail feature modulation 수행 |
| Dual-device protocol | CPU no-GPU 조건과 AWS CUDA 조건을 분리해 평가 |

## 공개용 패키지 범위

포함:

- LiteRaceSegNet 구현 코드
- 학습/추론/비교 스크립트
- GitHub README용 구조도 및 시각자료
- static demo page
- baseline/evidence template
- smoke check script

제외:

- 논문 PDF/DOCX
- `paper/` 폴더
- raw dataset, mask, private image
- checkpoint, pretrained weight
- API key, credential

## GitHub에서 강조할 포인트

1. “작은 모델”만 강조하지 말고, boundary degradation 문제를 먼저 설명합니다.
2. final 수치가 없는 항목은 TBD로 두고, 없는 수치를 만들지 않습니다.
3. 구조도에서는 `custom LiteIR backbone`이라고 표현합니다.
4. CPU latency와 GPU latency는 서로 직접 비교하지 않고 device condition별로 해석합니다.
5. segmentation prediction path와 관련 없는 보조 UI/문서 자료는 README 첫 화면에서 제외합니다.
