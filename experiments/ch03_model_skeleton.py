"""Run against the pinned upstream code, without downloading weights."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('MINIMIND_SOURCE_ROOT', ROOT.parent / 'minimind_resources')) / 'minimind'
EXPECTED = '233d00fa068ec2b8c0d4f07b703934d6391a5a73'


def run():
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED:
        raise RuntimeError(f'Source mismatch: {commit}; expected {EXPECTED}')
    source_file = SOURCE / 'model/model_minimind.py'
    if subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--', 'model/model_minimind.py'], text=True).strip():
        raise RuntimeError('The model source has uncommitted changes')
    sys.path.insert(0, str(SOURCE))
    import torch
    import transformers
    from model.model_minimind import RMSNorm, FeedForward, MiniMindConfig, MiniMindForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(42)
    results = {}
    norm = RMSNorm(2, eps=1e-6)
    x = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    y = norm(x)
    expected = x / (torch.tensor(12.5) + 1e-6).sqrt()
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(y.square().mean(-1), torch.ones(1, 1), rtol=1e-5, atol=1e-6)
    y.sum().backward()
    for grad in (x.grad, norm.weight.grad):
        assert grad is not None and torch.isfinite(grad).all()
    assert x.grad.abs().sum() > 0 and norm.weight.grad.abs().sum() > 0
    results['hand_calculation'] = y.detach().flatten().tolist()
    results['input_gradient'] = x.grad.flatten().tolist()
    zero = torch.zeros(1, 1, 2, requires_grad=True)
    zero_out = norm(zero)
    assert torch.equal(zero_out, torch.zeros_like(zero_out))
    zero_out.sum().backward()
    assert torch.isfinite(zero.grad).all()
    results['zero_input'] = 'finite forward and backward; zero output'
    with torch.no_grad():
        norm.weight.fill_(2)
    scaled = norm(x.detach())
    torch.testing.assert_close(scaled, y.detach() * 2)
    results['learned_scale'] = float(scaled.square().mean().sqrt())
    large = torch.full((1, 1, 2), 300.0, dtype=torch.float16)
    assert torch.isinf(large.square()).all()
    stable = norm(large)
    assert stable.dtype == large.dtype and torch.isfinite(stable).all()
    results['fp16_square_overflow'] = 'naive square inf; upstream norm finite'

    config = MiniMindConfig(hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        intermediate_size=64, vocab_size=32, max_position_embeddings=64,
        use_moe=False, flash_attn=False, dropout=0.0)
    ff = FeedForward(config)
    hidden = torch.randn(2, 4, 32)
    reference = ff.down_proj(torch.nn.functional.silu(ff.gate_proj(hidden)) * ff.up_proj(hidden))
    torch.testing.assert_close(ff(hidden), reference)
    assert ff(hidden).shape == hidden.shape
    results['ffn_parameters'] = sum(p.numel() for p in ff.parameters())
    model = MiniMindForCausalLM(config)
    assert model.model.embed_tokens.weight is model.lm_head.weight
    ids = torch.tensor([[1, 4, 7, 2]])
    output = model(ids, labels=ids, use_cache=True)
    assert output.logits.shape == (1, 4, 32)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()
    results['model_logits_shape'] = list(output.logits.shape)
    results['model_parameters'] = sum(p.numel() for p in model.parameters())
    results['weight_tying'] = 'same Parameter object'
    results['loss'] = output.loss.item()
    report = dict(source_commit=commit, source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
        platform=platform.platform(), device='cpu', seed=42, results=results, status='passed')
    target = ROOT / 'build/experiments/ch03-results.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
