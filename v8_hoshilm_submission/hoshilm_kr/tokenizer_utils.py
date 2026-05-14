from __future__ import annotations

from pathlib import Path
from typing import List


class CharTokenizer:
    def __init__(self, text: str | None = None, vocab_path: str | None = None):
        if vocab_path and Path(vocab_path).exists():
            chars = Path(vocab_path).read_text(encoding="utf-8").splitlines()
        else:
            if text is None:
                raise ValueError("Provide text or vocab_path for CharTokenizer")
            chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)

    def save(self, vocab_path: str) -> None:
        Path(vocab_path).write_text("\n".join(self.itos[i] for i in range(self.vocab_size)), encoding="utf-8")


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError("sentencepiece is required. Install with: pip install sentencepiece") from e
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.vocab_size()

    def encode(self, s: str) -> List[int]:
        return self.sp.encode_as_ids(s)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode_ids([int(i) for i in ids])
