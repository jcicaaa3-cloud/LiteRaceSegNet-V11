# Reviewer quick start

1. `README.md`를 먼저 봅니다.
2. 브라우저에서 `index.html`을 열어 구조와 시각자료를 확인합니다.
3. 코드 검증은 아래 명령으로 시작합니다.

```bash
pip install -r requirements.txt
python scripts/smoke_check_literace.py
```

데이터셋과 checkpoint는 공개 zip에 포함되지 않으므로 training/evaluation은 사용 권한이 있는 데이터셋을 배치한 뒤 실행해야 합니다.
