# Model Card: HoshiLM-KR Lab

## Intended use

Educational reproduction of a decoder-only Transformer decoder-only language model pipeline.

## Non-goals

- Not a production LLM
- Not a commercial chatbot clone
- Not trained on a sufficiently large corpus for general language understanding
- Not intended for public deployment as an open chatbot

## Architecture

- Decoder-only Transformer
- Causal self-attention
- Learned token embeddings
- Learned positional embeddings
- MLP feed-forward blocks
- LayerNorm and residual connections
- Next-token prediction objective

## Data

The current package includes a small Korean dummy corpus. It is suitable for smoke tests and overfitting checks, not for broad language capability.

## Known limitations

- Small data causes memorization
- Generated text may be repetitive or incoherent
- XL config needs larger corpus and longer training
- No alignment, safety tuning, retrieval, or instruction tuning

## Recommended presentation wording

> HoshiLM-KR is a toy-scale decoder-only Transformer implementation for reproducing the core decoder-only Transformer training pipeline. It demonstrates tokenization, causal self-attention, next-token prediction, checkpointing, and sampling on a small Korean corpus.

