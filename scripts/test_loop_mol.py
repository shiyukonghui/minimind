# Loop-MoL-MiniMind 冒烟测试：形状 / 冻结 / 梯度流 / 损失组成 / 底座载入
# 用法：python scripts/test_loop_mol.py
import sys, math
import torch

sys.path.insert(0, r'F:\MachineLearn\minimind')
from model.model_loop_mol import LoopMiniMindForCausalLM, MoLFFN, JevHeads, LoopedBlock, jev_calibration_losses
from model.model_minimind import MiniMindConfig

CKPT = r'F:\MachineLearn\minimind\checkpoints\pretrain_512.pth'

def test_shapes_and_freeze():
    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                         num_key_value_heads=2, vocab_size=6400)
    m = LoopMiniMindForCausalLM(cfg, k_max=32, top_k=2, rank=16, max_steps=8)
    ids = torch.randint(3, 6400, (2, 16))
    out = m(ids, T=4, labels=ids)
    assert out['logits'].shape == (2, 16, 6400)
    assert out['hidden'].shape == (2, 16, 512)
    assert out['halt_w'].shape == (2, 16, 4)
    # halt 权重每步归一化且和为 1
    assert torch.allclose(out['halt_w'].sum(-1), torch.ones(2, 16), atol=1e-5)
    for key in ('lm_loss', 'aux_loss', 'depth_ent', 'abstain_brier', 'loss'):
        assert key in out, key
        assert torch.isfinite(out[key]), key
    print('[1] 形状/损失 ✓  loss=%.3f lm=%.3f aux=%.3f 平均深度=%.2f' % (
        out['loss'], out['lm_loss'], out['aux_loss'], out['mean_depth']))

    # 冻结检查：底座参数 requires_grad=False，可训练组件=True
    frozen = sum(1 for p in m.parameters() if not p.requires_grad)
    train = sum(1 for p in m.parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in m.embed_tokens.parameters())
    assert all(not p.requires_grad for p in m.block.attn.parameters())
    assert all(not p.requires_grad for p in m.block.mol.shared.parameters())
    assert all(p.requires_grad for p in m.block.jev.parameters())
    assert all(p.requires_grad for p in m.block.mol.experts.parameters())
    print(f'[2] 冻结策略 ✓  可训练张量 {train} 个 / 冻结 {frozen} 个，可训练参数 {m.trainable_param_count:,}')

    # 梯度流：loss 反传后专家/决策头有梯度，底座无
    out['loss'].backward()
    g_expert = m.block.mol.experts[0].B.weight.grad
    g_halt = m.block.jev.halt.weight.grad
    g_attn = m.block.attn.q_proj.weight.grad
    assert g_expert is not None and g_expert.abs().sum() > 0
    assert g_halt is not None and g_halt.abs().sum() > 0
    assert g_attn is None or g_attn.abs().sum() == 0
    print('[3] 梯度流 ✓  专家/决策头有梯度，冻结底座无梯度')

def test_load_base():
    import os
    if not os.path.exists(CKPT):
        print('[4] 跳过底座载入(无 checkpoint)')
        return
    sd = torch.load(CKPT, map_location='cpu')
    sd = sd.get('model_state', sd) if isinstance(sd, dict) and 'model_state' in sd else sd
    keys = list(sd.keys())
    assert any('model.layers.3.mlp' in k for k in keys), 'checkpoint 键格式异常'
    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                         num_key_value_heads=2, vocab_size=6400)
    m = LoopMiniMindForCausalLM(cfg)
    loaded, res = m.load_base(CKPT, layer_id=3)
    n_layer3 = sum(1 for k in sd if 'model.layers.3.' in k)
    print(f'[4] 底座载入 ✓  映射 {len(loaded)} 个张量(层3 共 {n_layer3})，unexpected={len(res.unexpected_keys)}')
    # 验证 attn 权重确实载入(非随机)；ckpt 为 fp16，模型侧统一转 float32
    assert torch.allclose(m.block.attn.q_proj.weight, sd['model.layers.3.self_attn.q_proj.weight'].float())
    assert torch.allclose(m.block.mol.shared.gate_proj.weight, sd['model.layers.3.mlp.gate_proj.weight'].float())
    print('[5] 权重一致性 ✓  q_proj/gate_proj 与 checkpoint 完全相等')

def test_timestep_zero_start():
    """零初始化 timestep 下，循环第 0 步输出应接近底座单层(专家 B=0 中性)。"""
    cfg = MiniMindConfig(hidden_size=512, num_attention_heads=8, num_key_value_heads=2, vocab_size=6400)
    m = LoopMiniMindForCausalLM(cfg)
    m.eval()
    ids = torch.randint(3, 6400, (1, 8))
    with torch.no_grad():
        h = m.embed_tokens(ids)
        pos = (m.freqs_cos[:8], m.freqs_sin[:8])
        # 第 0 步 alpha=1，专家 B 零初始化 -> 输出 = x + attn + shared_ffn
        a, _ = m.block.attn(m.block.norm1(h), pos, None, False, None)
        x1 = h + a
        f0 = m.block.mol.shared(m.block.norm2(x1))
        ref = x1 + f0
        out = m.block(h, pos, T=1)['hidden']
        assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()
    print('[6] 中性起点 ✓  T=1 时循环块输出与底座单层完全一致')

if __name__ == '__main__':
    torch.manual_seed(0)
    test_shapes_and_freeze()
    test_load_base()
    test_timestep_zero_start()
    print('\n全部通过')
