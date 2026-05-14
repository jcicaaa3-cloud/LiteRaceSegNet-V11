from __future__ import annotations
import argparse
from project_qa_engine import ProjectQABot

def main():
    ap = argparse.ArgumentParser(description='LiteRaceSegNet/HoshiLM evidence-grounded QA chat')
    ap.add_argument('--no-lm', action='store_true')
    ap.add_argument('--ckpt', default=None)
    args = ap.parse_args()
    bot = ProjectQABot(ckpt=args.ckpt)
    print('HoshiLM Project QA. 종료하려면 exit 입력.')
    print('mode:', 'evidence+hoshilm' if not args.no_lm and bot.ckpt else 'evidence-only')
    while True:
        q = input('\n질문> ').strip()
        if q.lower() in {'exit','quit','q'}:
            break
        if not q:
            continue
        res = bot.ask(q, use_lm=not args.no_lm)
        print('\n답변:', res['answer'])
        if res.get('model_note'):
            print('\nHoshiLM 생성 보조:', res['model_note'])
        if res.get('sources'):
            print('\n근거:', ', '.join(res['sources']))
if __name__ == '__main__':
    main()
