# LiteRaceSegNet-V11

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-segmentation-EE4C2C)
![License](https://img.shields.io/badge/license-portfolio%20viewing%20only-64748B)
![Status](https://img.shields.io/badge/results-TBD%20safe-D97706)
![Paper](https://img.shields.io/badge/manuscript-not%20included-475569)

<p align="center">
  <img src="docs/assets/readme_hero.png" alt="LiteRaceSegNet GitHub visual overview" width="100%" />
</p>

**LiteRaceSegNet-V11**은 도로 손상 semantic segmentation 프로젝트를 GitHub 포트폴리오 형태로 보여주기 위한 공개 패키지입니다. 핵심 질문은 단순합니다. **작고 불규칙한 도로 손상 mask와 경계를 경량 CNN 구조에서 얼마나 안정적으로 보존할 수 있는가?**

이 저장소에는 segmentation source code, static demo, 구조도, 실험 evidence template, 실행 script, 그리고 실험 근거를 설명하기 위한 선택형 HoshiLM Project QA module이 포함됩니다. 논문 PDF/DOCX, private dataset, raw mask, checkpoint, pretrained weight, credential은 공개 패키지에서 제외했습니다.

> 추천 저장소 설명  
> `Lightweight road-damage segmentation with a boundary-guided dual-branch CNN and CPU/GPU evidence pipeline.`

---

## 1. 한눈에 보는 구조

LiteRaceSegNet은 detail branch, context branch, LiteASPP, boundary auxiliary head, boundary-guided fusion을 결합한 경량 segmentation 구조입니다.

<p align="center">
  <img src="docs/github_assets/literacesegnet_architecture_overview_corrected.png" alt="LiteRaceSegNet corrected architecture overview" width="100%" />
</p>

GitHub에 노출되는 주요 diagram은 **직접 그린 텍스트/도형 기반 이미지**로 다시 만들었습니다. 저작권/상표권이 애매할 수 있는 장식 아이콘은 제거했고, backbone 표기는 실제 구현과 맞게 **Custom LiteIR-style Backbone**으로 보정했습니다.

---

## 2. 프로젝트 포지션

이 저장소는 “SOTA 달성”을 과장하는 프로젝트가 아니라, 아래 네 가지를 분명히 보여주는 포트폴리오/연구 시연용 저장소입니다.

| 축 | 보여주는 내용 | 의미 |
|---|---|---|
| 도로 손상 segmentation | 포트홀, 균열, 표면 파손 mask 예측 | 도로 손상은 작고, 낮은 대비이며, 배경 도로 질감과 비슷함 |
| Boundary degradation | boundary erosion, dilation, thin-structure disappearance, context absorption | mIoU만으로는 손상 경계 품질을 충분히 설명하기 어려움 |
| 경량 모델 설계 | detail branch + context branch + LiteASPP + boundary-guided fusion | 작은 모델에서 경계/문맥 trade-off를 검증 |
| Evidence pipeline | CPU evidence, GPU evidence, dual-device summary, ablation template | 측정된 값과 TBD 값을 분리해 없는 수치를 만들지 않음 |
| Project QA | HoshiLM Project QA가 실험 근거와 프로젝트 사실을 설명 | segmentation mask를 바꾸는 모델이 아니라 설명/보고 지원 모듈 |

---

## 3. Architecture and output contract

<p align="center">
  <img src="docs/assets/literacesegnet_architecture_github_clean.png" alt="Clean LiteRaceSegNet architecture" width="100%" />
</p>

핵심 구현 파일은 `seg/core/lightweight_race.py`입니다. dataset 없이 smoke check를 실행하면 모델 forward contract를 확인할 수 있습니다.

| Output | Smoke check shape | 역할 |
|---|---:|---|
| `out` | `(B, 2, H, W)` | main segmentation logits |
| `aux` | `(B, 2, H, W)` | auxiliary segmentation output |
| `boundary` | `(B, 1, H, W)` | boundary-aware supervision/fusion 분석용 boundary logit |

---

## 4. 현재 reference evidence

아래 값은 작은 validation run에서 나온 reference evidence입니다. 최종 일반화 성능 주장으로 쓰면 안 되고, V11 official evidence script를 돌린 뒤 최종표를 채우는 방식이 맞습니다.

| Model | Params | FP32 parameter size | Pixel Acc. | binary mIoU | Damage IoU | 상태 |
|---|---:|---:|---:|---:|---:|---|
| LiteRaceSegNet reference | `0.1245M` | `0.475 MiB` | `0.9157` | `0.7988` | `0.7029` | preliminary reference |

<p align="center">
  <img src="docs/assets/evidence_status_matrix.png" alt="Evidence status matrix" width="92%" />
</p>

SegFormer/DeepLab/LRASPP baseline, Boundary IoU, CPU/GPU latency, robustness, ablation 수치는 아직 `TBD`로 두었습니다. 없는 숫자를 만들지 않고, 실행 후 evidence table에 채우는 구조입니다.

---

## 5. CPU/GPU evaluation protocol

CPU와 GPU 수치는 서로 다른 질문에 답합니다. CPU latency와 GPU latency를 절대값으로 직접 비교하지 않습니다.

<p align="center">
  <img src="docs/assets/evaluation_protocol_github_clean.png" alt="Dual-device evaluation protocol" width="92%" />
</p>

| 조건 | 목적 | 주요 출력 |
|---|---|---|
| CPU / no-GPU | 현장형 추론 가능성 | latency, FPS, FP32 size, parameter count, Damage IoU |
| AWS GPU / CUDA | 가속 추론과 대량 처리 가능성 | latency, throughput, CUDA memory, AMP, batch size |
| Dual-device synthesis | 보고서/README용 종합 비교 | `final_evidence/` 아래 CSV, JSON, Markdown summary |

---

## 6. HoshiLM Project QA는 왜 다시 넣었나

HoshiLM Project QA는 포트폴리오 가시성 측면에서 장점이 있습니다. 다만 segmentation model과 섞어 보이면 오해가 생기므로, 이번 버전에서는 **optional reporting/support module**로 명확히 분리했습니다.

**시각자료 저작권 안전 패치.** Project QA flow diagram의 기존 sparkle 계열 장식 표식은 제거했고, 기본 도형으로 그린 text-only `QA` badge로 교체했습니다. 공개 패키지에는 제3자 decorative logo/icon으로 오해될 수 있는 표식을 사용하지 않습니다.


<p align="center">
  <img src="docs/github_assets/hoshilm_project_qa_flow_diagram.png" alt="HoshiLM Project QA flow" width="100%" />
</p>

정적 미리보기 위치:

```text
v8_hoshilm_submission/web_demo/
v8_hoshilm_submission/web_project_qa/
```

GitHub Pages에서는 UI preview를 볼 수 있고, 실제 질의응답은 local/AWS Python API가 필요합니다.

---

## 7. Repository structure

<p align="center">
  <img src="docs/github_assets/repo_structure.png" alt="Repository structure for GitHub" width="100%" />
</p>

```text
.
├─ README.md
├─ README_KO.md
├─ LICENSE
├─ index.html
├─ seg/                               # LiteRaceSegNet, training, inference, metrics
├─ scripts/                           # Linux/AWS scripts + smoke check
├─ docs/
│  ├─ assets/                          # README용 diagram, visual evidence
│  └─ github_assets/                   # 포트폴리오 flow diagram, upload visual
├─ demo/                               # static segmentation demo preview
├─ v8_hoshilm_submission/              # optional Project QA/static demo extension
├─ evidence_templates/                 # 결과표 입력 template
├─ datasets/                           # 빈 layout만 포함, 실제 data 없음
└─ final_evidence/                     # 생성 결과 위치, .gitkeep만 추적
```

---

## 8. Quick start

### 설치

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

선택 dependency:

```bash
pip install -r requirements_transformer_optional.txt
pip install -r requirements_service.txt
```

### Dataset 없이 smoke check

```bash
python scripts/smoke_check_literace.py
```

예상 출력:

```text
LiteRaceSegNet smoke check OK
trainable_params=124,509
outputs={'out': (1, 2, 64, 64), 'aux': (1, 2, 64, 64), 'boundary': (1, 1, 64, 64)}
```

### Dataset 배치

이 공개 패키지에는 dataset이 포함되지 않습니다. 사용할 권한이 있는 image-mask pair만 아래 구조에 넣어야 합니다.

```text
datasets/pothole_binary/processed/
├─ train/
│  ├─ images/
│  └─ masks/
├─ val/
│  ├─ images/
│  └─ masks/
└─ test/
   ├─ images/
   └─ masks/
```

---

## 9. GitHub upload flow

<p align="center">
  <img src="docs/github_assets/github_upload_flow.png" alt="GitHub upload flow" width="100%" />
</p>

```bash
git init
git branch -M main
git add .
git commit -m "Initial LiteRaceSegNet V11 visual release"
git remote add origin https://github.com/jcicaaa3-cloud/LiteRaceSegNet-V11.git
git push -u origin main
```

---

## 10. GitHub Pages 주의점

이번 `index.html`에서는 `./README.md`로 이동하지 않게 고쳤습니다. GitHub Pages에서 Markdown 파일로 바로 이동하면 README가 GitHub처럼 예쁘게 렌더링되지 않고 이상하게 보일 수 있기 때문입니다. 그래서 Pages 버튼은 GitHub repository의 README URL로 이동하도록 바꿨습니다.

Pages 설정:

```text
Settings → Pages → Deploy from a branch → main → /root
```

Pages 주소:

```text
https://jcicaaa3-cloud.github.io/LiteRaceSegNet-V11/
```

---

## 11. 공개 범위

포함:

- LiteRaceSegNet code/configs
- segmentation/evaluation scripts
- static demo pages
- Project QA preview/support code
- documentation, diagrams, templates, smoke check

제외:

- 논문 PDF/DOCX
- raw dataset, private image, mask
- checkpoint, pretrained weight, generated run
- `.env`, API key, cloud credential, `.pem` file

---

## 12. 라이선스 및 사용 제한

Copyright (c) 2026 김원석

본 프로젝트의 LiteRaceSegNet 관련 코드, 문서, 구조도, 실험 기록, 설정 파일은 포트폴리오 및 학업적 시연 목적으로만 공개됩니다.

작성자의 사전 허가 없이 본 프로젝트 또는 그 구성 요소를 복제, 재배포, 수정, 공개, 2차 저작물 제작, 상업적 목적으로 사용하는 것을 허용하지 않습니다.

단, 본 프로젝트에서 참조하거나 사용하는 외부 라이브러리, 프레임워크, 데이터셋, 모델 구현체는 각 원 저작자와 해당 라이선스 조건을 따릅니다.

자세한 내용은 [`LICENSE`](LICENSE), [`NOTICE.txt`](NOTICE.txt), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)를 확인하십시오.
