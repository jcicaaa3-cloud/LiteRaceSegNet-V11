# GitHub Pages README Link Fix

기존 `index.html`에서 README 버튼이 `./README.md` 또는 `./README_KO.md`를 향하면 GitHub Pages에서는 Markdown이 GitHub 저장소 화면처럼 렌더링되지 않을 수 있습니다. 이 때문에 페이지가 이상하게 보일 수 있습니다.

이번 버전에서는 아래처럼 수정했습니다.

```html
<a class="btn" href="https://github.com/jcicaaa3-cloud/LiteRaceSegNet-V11#readme" target="_blank" rel="noopener">View README on GitHub</a>
<a class="btn secondary" href="https://github.com/jcicaaa3-cloud/LiteRaceSegNet-V11/blob/main/README_KO.md" target="_blank" rel="noopener">Korean README</a>
```

즉, GitHub Pages는 project landing page 역할을 하고, README는 GitHub repository 화면에서 보게 합니다.
