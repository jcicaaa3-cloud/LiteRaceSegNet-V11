# GitHub upload guide

## 추천 저장소 설명

```text
Lightweight road-damage segmentation with a boundary-guided dual-branch CNN and CPU/GPU evidence pipeline.
```

## 추천 topics

```text
semantic-segmentation, road-damage-segmentation, pothole-detection, lightweight-cnn, boundary-aware, pytorch, edge-ai, computer-vision
```

## 업로드 전 확인

```bash
python scripts/smoke_check_literace.py
```

정상 출력 예시:

```text
LiteRaceSegNet smoke check OK
trainable_params=124,509
outputs={'out': (1, 2, 64, 64), 'aux': (1, 2, 64, 64), 'boundary': (1, 1, 64, 64)}
```

공개 금지 파일 확인:

```bash
find . -iname "*.pdf" -o -iname "*.docx"
find . -type d -iname "paper"
find . -type f \( -iname "*.pt" -o -iname "*.pth" -o -iname "*.ckpt" \)
```

## 새 저장소에 올리기

```bash
git init
git add .
git commit -m "Initial LiteRaceSegNet GitHub visual release"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

## 기존 저장소에 논문 파일 제거 후 올리기

```bash
git rm -r --ignore-unmatch paper
git rm --ignore-unmatch *.pdf *.docx
git add .
git commit -m "Convert to public-safe GitHub visual release"
git push
```

## GitHub Pages 설정

1. `Settings → Pages`
2. Branch: `main`
3. Folder: `/root`
4. Pages URL에서 `index.html`이 뜨는지 확인

## README 이미지 구성

GitHub 첫 화면 이미지는 아래 순서로 유지하는 것을 권장합니다.

1. `docs/assets/readme_hero.png`
2. `docs/assets/literacesegnet_architecture_github_clean.png`
3. `docs/assets/evidence_status_matrix.png`
4. `docs/assets/visual_evidence_gallery.png`
5. `docs/assets/evaluation_protocol_github_clean.png`

기존 시각 자료는 `docs/github_assets/`에 보존되어 있습니다. 단, README 첫 화면은 corrected backbone wording이 적용된 새 이미지를 사용하는 것이 안전합니다.
