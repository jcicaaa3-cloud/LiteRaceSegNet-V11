# Asset and copyright policy

This repository is prepared for public portfolio use. Keep it clean before pushing to GitHub.

## Safe to commit

- source code written for this project
- configuration files
- small placeholder files such as `.gitkeep`
- documentation
- architecture diagrams authored for this project
- result templates without real private data

## Do not commit

- raw road images
- dataset masks or labels
- downloaded public datasets
- paid or restricted datasets
- private camera footage
- trained checkpoints
- pretrained model weights
- Hugging Face cache folders
- generated overlays if the input image license is unclear
- `.env`, API keys, cloud credentials
- private manuscript DOCX/PDF drafts containing personal information or school formatting; not included in this GitHub release

## Repository copyright scope

`LICENSE` and `NOTICE.txt` cover LiteRaceSegNet-related code, documentation, diagrams, experiment notes, configuration files, and scripts authored for this portfolio project.

This repository is not released as an MIT-licensed open-source project. It is public for portfolio viewing and academic demonstration only.

작성자의 사전 허가 없이 본 프로젝트 또는 그 구성 요소를 복제, 재배포, 수정, 공개, 2차 저작물 제작, 상업적 목적으로 사용하는 것을 허용하지 않습니다.

External third-party packages, model weights, datasets, APIs, and referenced model implementations are not included and are not relicensed.

## Dataset wording for README or resume

Use this wording:

> The repository does not include datasets or weights. Experiments require the user to place permitted image-mask pairs under the documented dataset layout.

Avoid this wording:

> Dataset is included.
> We provide pretrained weights.
> Anyone can freely use all assets in this repository.

## Result image policy

Only publish overlay images when the source road image can be redistributed. If the license is unclear, keep overlays local and publish only aggregate metrics.

## License summary

This repository is public for portfolio viewing and academic demonstration only. It is not released under a permissive open-source license. See `LICENSE` and `NOTICE.txt`.

Copyright (c) 2026 김원석. Unauthorized copying, redistribution, modification, public reposting, derivative work, or commercial use of LiteRaceSegNet-related code, documentation, diagrams, experiment records, configuration files, and repository assets is not permitted without prior written permission from the author.

Third-party libraries, frameworks, datasets, and model implementations remain governed by their original licenses and are not relicensed by this repository.


## Visual asset safety update

The public GitHub package should not depend on third-party decorative logo or icon marks. The previous sparkle-style mark in the Project QA flow diagram was replaced by a custom text-only `QA` badge drawn from basic shapes. This avoids unnecessary copyright/trademark ambiguity while preserving the visual explanation flow.


## Decorative icon policy

The GitHub-facing README/Page diagrams in this release are intentionally generated from simple text, rectangles, lines, arrows, and basic geometric shapes authored for this project. Ambiguous decorative icon marks, third-party icon-pack graphics, and logo-like sparkle marks were removed from the public-facing images.

External trademarks such as Python, PyTorch, GitHub, AWS, and Hugging Face may be mentioned only as textual references to tools or services. Such names remain the property of their respective owners and are not claimed as project assets.
