from __future__ import annotations
import argparse
try:
    import yaml
except ImportError:
    yaml = None
from model import LMConfig, HoshiLM, count_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/hoshilm_m.yaml')
    args = parser.parse_args()
    if yaml is None:
        raise SystemExit('Install pyyaml: pip install pyyaml')
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    vocab_size = int(cfg.get('vocab_size_override') or cfg.get('expected_vocab_size') or 8000)
    lm_cfg = LMConfig(
        vocab_size=vocab_size,
        block_size=int(cfg['block_size']),
        n_layer=int(cfg['n_layer']),
        n_head=int(cfg['n_head']),
        n_embd=int(cfg['n_embd']),
        dropout=float(cfg.get('dropout', 0.1)),
        bias=bool(cfg.get('bias', True)),
        tie_weights=bool(cfg.get('tie_weights', True)),
    )
    model = HoshiLM(lm_cfg)
    params = count_parameters(model)
    print(f'config={args.config}')
    print(f'vocab_size={vocab_size}')
    print(f'params={params:,}')
    print(f'params_m={params/1e6:.3f}M')
    print(f'fp32_param_size_mib={params*4/1024/1024:.2f} MiB')


if __name__ == '__main__':
    main()
