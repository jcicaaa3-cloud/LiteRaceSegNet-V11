from __future__ import annotations
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/data.txt')
    parser.add_argument('--model_prefix', default='tokenizer/sp')
    parser.add_argument('--vocab_size', type=int, default=4000)
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ImportError as e:
        raise SystemExit('sentencepiece is required. Install with: pip install sentencepiece') from e

    Path(args.model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=0.9995,
        model_type='bpe',
    )
    print(f'DONE: {args.model_prefix}.model / {args.model_prefix}.vocab')


if __name__ == '__main__':
    main()
