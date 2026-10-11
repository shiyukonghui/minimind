# 从现有 SFT 数据抽取 user 问题作为蒸馏种子集（任务⑧前置）
# 用法：python scripts/make_distill_seeds.py [--src dataset/sft_t2t_mini.jsonl] [--out dataset/distill_seeds.jsonl] [--n 200]
import json, argparse, random

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=r'F:\MachineLearn\minimind\dataset\sft_t2t_mini.jsonl')
ap.add_argument('--out', default=r'F:\MachineLearn\minimind\dataset\distill_seeds.jsonl')
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()

random.seed(a.seed)
prompts = []
seen = set()
with open(a.src, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        cs = json.loads(line)['conversations']
        if cs and cs[0]['role'] == 'user':
            q = cs[0]['content'].strip()
            # 过滤过短/过长问题，保证蒸馏种子质量
            if 4 <= len(q) <= 200 and q not in seen:
                seen.add(q)
                prompts.append(q)
random.shuffle(prompts)
prompts = prompts[:a.n]
with open(a.out, 'w', encoding='utf-8') as f:
    for q in prompts:
        f.write(json.dumps({'prompt': q}, ensure_ascii=False) + '\n')
print(f'抽取 {len(prompts)} 条种子 -> {a.out}')
