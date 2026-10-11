# 验证 E1-C/E1-D 初始化产物能被 MiniMind 严格加载并正常前向
import sys
import torch

sys.path.insert(0, r'F:\MachineLearn\minimind')
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                     num_key_value_heads=2, vocab_size=6400)
m = MiniMindForCausalLM(cfg)
key_q = 'model.layers.0.self_attn.q_proj.weight'
ids = torch.randint(3, 6400, (1, 16))
for tag in ('e1c_dist_sample', 'e1d_transplant'):
    path = rf'F:\MachineLearn\minimind\checkpoints\{tag}.pth'
    sd = torch.load(path, map_location='cpu')['model_state']
    m.load_state_dict(sd, strict=True)          # 严格加载：键必须完全匹配
    with torch.no_grad():
        out = m(ids)
    q = sd[key_q]
    print(f'{tag}: strict 加载 OK, logits{tuple(out.logits.shape)}, '
          f'finite={torch.isfinite(out.logits).all().item()}, '
          f'q_proj std={q.std():.4f} kurt≈{(((q - q.mean()) / q.std()) ** 4).mean() - 3:.2f}')
print('验证通过')
