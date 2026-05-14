from __future__ import annotations
import argparse
import torch
from model import LMConfig, HoshiLM
from tokenizer_utils import CharTokenizer, SentencePieceTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--prompt', default='프로젝트는')
    parser.add_argument('--tokens', type=int, default=120)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=40)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt['model_config']
    train_cfg = ckpt.get('train_config', {})

    if train_cfg.get('tokenizer') == 'sentencepiece':
        tok = SentencePieceTokenizer(train_cfg['sp_model_path'])
    else:
        tok = CharTokenizer(vocab_path=train_cfg['char_vocab_path'])

    model = HoshiLM(LMConfig(**cfg)).to(device)
    model.load_state_dict(ckpt['model'])
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, args.temperature, args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == '__main__':
    main()
