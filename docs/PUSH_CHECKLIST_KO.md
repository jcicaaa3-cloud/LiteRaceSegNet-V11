# GitHub push checklist

## 1. 공개 금지 파일 확인

```bash
find . -iname "*.pdf" -o -iname "*.docx"
find . -type d -iname "paper"
find . -type f \( -iname "*.pt" -o -iname "*.pth" -o -iname "*.ckpt" \)
```

위 명령에서 공개하면 안 되는 파일이 나오면 삭제하거나 `.gitignore`에 반영합니다.

## 2. smoke check

```bash
python scripts/smoke_check_literace.py
```

정상 출력:

```text
LiteRaceSegNet smoke check OK
trainable_params=124,509
```

## 3. README 이미지 링크 확인

README에서 사용하는 핵심 이미지:

- `docs/assets/readme_hero.png`
- `docs/assets/literacesegnet_architecture_github_clean.png`
- `docs/assets/evidence_status_matrix.png`
- `docs/assets/visual_evidence_gallery.png`
- `docs/assets/evaluation_protocol_github_clean.png`
- `docs/assets/repo_map_github_clean.png`

## 4. 업로드 명령

```bash
git init
git add .
git commit -m "Initial LiteRaceSegNet GitHub visual release"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

## 5. GitHub Pages

1. `Settings → Pages`
2. Branch: `main`
3. Folder: `/root`
4. `index.html`이 landing page로 표시되는지 확인합니다.
