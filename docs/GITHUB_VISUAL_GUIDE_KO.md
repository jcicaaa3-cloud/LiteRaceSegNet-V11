# GitHub Visual Guide

이 문서는 GitHub README와 GitHub Pages에서 어떤 이미지를 어디에 쓰는지 정리합니다.

## 메인 README / Pages 이미지

| 파일 | 사용 위치 | 목적 |
|---|---|---|
| `docs/assets/readme_hero.png` | README 상단 | 프로젝트 첫인상용 hero image |
| `docs/github_assets/literacesegnet_architecture_overview_corrected.png` | README/Pages architecture overview | 기존에 보기 좋았던 architecture overview를 유지하되 backbone 표기를 `Custom LiteIR-style Backbone`으로 보정한 이미지 |
| `docs/assets/literacesegnet_architecture_github_clean.png` | Architecture section | 구현 구조를 더 정확하게 설명하는 clean architecture diagram |
| `docs/assets/evidence_status_matrix.png` | Evidence section | 측정된 reference evidence와 TBD 항목 분리 |
| `docs/assets/evaluation_protocol_github_clean.png` | Evaluation section | CPU/GPU dual-device evaluation protocol |
| `docs/github_assets/hoshilm_project_qa_flow_diagram.png` | Optional Project QA section | HoshiLM Project QA가 segmentation model이 아니라 evidence/reporting support module임을 설명 |
| `docs/github_assets/repo_structure.png` | Repository structure section | GitHub 공개 패키지 구조 설명 |
| `docs/github_assets/github_upload_flow.png` | Upload flow section | command-line push 흐름 설명 |
| `docs/assets/visual_evidence_gallery.png` | Visual evidence section | validation/service-card 시각 증거 요약 |

## README에서 삭제하지 말아야 할 이미지

- `literacesegnet_architecture_overview_corrected.png`
- `hoshilm_project_qa_flow_diagram.png`
- `repo_structure.png`
- `github_upload_flow.png`

이 네 이미지는 포트폴리오 첫인상을 살리는 데 도움이 됩니다. 단, HoshiLM은 반드시 “optional reporting/support module”로 설명해야 합니다. segmentation mask를 생성하거나 성능을 개선하는 모델처럼 보이면 안 됩니다.

## GitHub Pages README 링크 주의

GitHub Pages에서 `./README.md`로 연결하면 GitHub 저장소 화면처럼 Markdown이 예쁘게 렌더링되지 않을 수 있습니다. 따라서 `index.html`의 README 버튼은 상대경로가 아니라 아래처럼 GitHub 저장소 README로 연결합니다.

```html
<a href="https://github.com/jcicaaa3-cloud/LiteRaceSegNet-V11#readme">View README on GitHub</a>
```

Korean README도 Pages 상대경로가 아니라 GitHub blob URL을 사용합니다.

```html
<a href="https://github.com/jcicaaa3-cloud/LiteRaceSegNet-V11/blob/main/README_KO.md">Korean README</a>
```

## 라이선스 표시

README, Pages footer, `LICENSE`, `NOTICE.txt`, `ASSET_AND_LICENSE_POLICY.md`에 모두 동일한 방향을 유지합니다.

- 공개 목적: 포트폴리오 및 학업적 시연
- 허용하지 않는 것: 무단 복제, 재배포, 수정, 공개, 2차 저작물, 상업적 사용
- 외부 라이브러리/데이터셋/모델: 원 저작자 및 원 라이선스 조건 따름
