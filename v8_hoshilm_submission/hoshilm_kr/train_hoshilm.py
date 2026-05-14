from __future__ import annotations

import argparse
import csv
import json
import math
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import random
import time

import torch

try:
    import yaml
except ImportError:
    yaml = None

from model import LMConfig, HoshiLM, count_parameters
from tokenizer_utils import CharTokenizer, SentencePieceTokenizer


def load_config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(('.yaml', '.yml')):
        if yaml is None:
            raise ImportError('pyyaml is required for YAML configs. Install with: pip install pyyaml')
        return yaml.safe_load(text)
    return json.loads(text)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device, ctx):
    model.eval()
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(cfg['eval_iters'])
        for k in range(cfg['eval_iters']):
            xb, yb = get_batch(data, cfg['block_size'], cfg['batch_size'], device)
            with ctx:
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/hoshilm_m.yaml')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get('seed', 42)))

    out_dir = Path(cfg.get('out_dir', 'runs/hoshilm'))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg['data_path'])
    text = data_path.read_text(encoding='utf-8')

    tokenizer_type = cfg.get('tokenizer', 'char')
    if tokenizer_type == 'sentencepiece':
        tok = SentencePieceTokenizer(cfg['sp_model_path'])
    else:
        vocab_path = out_dir / 'char_vocab.txt'
        tok = CharTokenizer(text=text)
        tok.save(str(vocab_path))
        cfg['char_vocab_path'] = str(vocab_path)

    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    split_idx = int(len(ids) * float(cfg.get('train_ratio', 0.9)))
    train_data, val_data = ids[:split_idx], ids[split_idx:]
    if len(val_data) <= cfg['block_size'] + 1 or len(train_data) <= cfg['block_size'] + 1:
        raise ValueError('Dataset is too small for this block_size. Lower block_size or add more text.')

    device = 'cuda' if torch.cuda.is_available() and cfg.get('device', 'auto') != 'cpu' else 'cpu'
    dtype = cfg.get('dtype', 'bfloat16') if device == 'cuda' else 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype) if device == 'cuda' else nullcontext()

    lm_cfg = LMConfig(
        vocab_size=tok.vocab_size,
        block_size=int(cfg['block_size']),
        n_layer=int(cfg['n_layer']),
        n_head=int(cfg['n_head']),
        n_embd=int(cfg['n_embd']),
        dropout=float(cfg.get('dropout', 0.1)),
        bias=bool(cfg.get('bias', True)),
        tie_weights=bool(cfg.get('tie_weights', True)),
    )
    model = HoshiLM(lm_cfg).to(device)
    n_params = count_parameters(model)
    print(f'device={device}, dtype={dtype}')
    print(f'vocab_size={tok.vocab_size}, train_tokens={len(train_data):,}, val_tokens={len(val_data):,}')
    print(f'params={n_params:,} ({n_params/1e6:.3f}M)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg.get('weight_decay', 0.1)), betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda' and dtype == 'float16'))

    start_iter = 0
    best_val = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt.get('iter_num', 0) + 1
        best_val = ckpt.get('best_val_loss', best_val)

    log_path = out_dir / 'train_log.csv'
    if not log_path.exists():
        with log_path.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['iter', 'train_loss', 'val_loss', 'lr', 'params_m', 'elapsed_sec'])

    max_iters = int(cfg['max_iters'])
    grad_accum = int(cfg.get('gradient_accumulation_steps', 1))
    eval_interval = int(cfg.get('eval_interval', 200))
    save_interval = int(cfg.get('save_interval', eval_interval))
    t0 = time.time()

    model.train()
    for iter_num in range(start_iter, max_iters):
        lr = float(cfg['learning_rate'])
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            xb, yb = get_batch(train_data, cfg['block_size'], cfg['batch_size'], device)
            with ctx:
                _, loss = model(xb, yb)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get('grad_clip', 1.0)))
        scaler.step(optimizer)
        scaler.update()

        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, cfg, device, ctx)
            elapsed = time.time() - t0
            print(f"iter={iter_num:6d} train={losses['train']:.4f} val={losses['val']:.4f} lr={lr:g} elapsed={elapsed:.1f}s")
            with log_path.open('a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([iter_num, losses['train'], losses['val'], lr, n_params/1e6, elapsed])

            if losses['val'] < best_val:
                best_val = losses['val']
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_config': asdict(lm_cfg), 'train_config': cfg, 'iter_num': iter_num, 'best_val_loss': best_val}, out_dir / 'best.pt')

        if iter_num % save_interval == 0 or iter_num == max_iters - 1:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_config': asdict(lm_cfg), 'train_config': cfg, 'iter_num': iter_num, 'best_val_loss': best_val}, out_dir / 'last.pt')

    # sample generation
    prompt = cfg.get('sample_prompt', '프로젝트는')
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=int(cfg.get('sample_tokens', 120)), temperature=float(cfg.get('temperature', 0.8)), top_k=int(cfg.get('top_k', 40)))
    sample = tok.decode(out[0].tolist())
    (out_dir / 'sample_output.txt').write_text(sample, encoding='utf-8')
    (out_dir / 'run_summary.json').write_text(json.dumps({'params': n_params, 'params_m': n_params/1e6, 'best_val_loss': best_val, 'config': cfg}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('sample saved:', out_dir / 'sample_output.txt')


if __name__ == '__main__':
    main()
