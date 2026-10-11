# 序列级蒸馏数据生成（设计文档 §3.7 RSI 阶段1，路线图任务⑧前置）
#
# 流程：种子问题集(jsonl, 含 prompt 字段) -> DeepSeek-V4.1-Flash 教师逐条作答
#       -> 输出 MiniMind SFT 格式 jsonl(conversations)，供 train_lora.py 直接训练。
#
# 用法：
#   python scripts/gen_distill_data.py --seeds dataset/distill_seeds.jsonl \
#       --out dataset/distill_v41_sft.jsonl [--limit 50]
# 无 DEEPSEEK_API_KEY 时自动 dry_run（占位回答，验证全链路但不产生真实蒸馏价值）。
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_client import TeacherClient

SYSTEM_PROMPT = '你是一个高效、准确的AI助手，回答简洁清晰。'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default=r'F:\MachineLearn\minimind\dataset\distill_seeds.jsonl')
    ap.add_argument('--out', default=r'F:\MachineLearn\minimind\dataset\distill_v41_sft.jsonl')
    ap.add_argument('--cache', default=r'F:\MachineLearn\minimind\dataset\distill_cache.jsonl')
    ap.add_argument('--limit', type=int, default=0, help='仅处理前 N 条(0=全部)')
    args = ap.parse_args()

    seeds = []
    with open(args.seeds, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))
    if args.limit:
        seeds = seeds[:args.limit]

    client = TeacherClient(cache_path=args.cache)
    mode = 'DRY_RUN(无API key，占位回答)' if client.dry_run else f'API({client.__class__.__name__})'
    print(f'种子 {len(seeds)} 条，模式：{mode}')

    done = 0
    with open(args.out, 'w', encoding='utf-8') as fout:
        for i, s in enumerate(seeds, 1):
            q = s.get('prompt') or s.get('question')
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': q}]
            try:
                ans = client.chat(messages)
            except Exception as e:
                print(f'[{i}] 失败跳过: {e}')
                continue
            rec = {'conversations': [
                {'role': 'user', 'content': q},
                {'role': 'assistant', 'content': ans},
            ]}
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            done += 1
            if i % 10 == 0 or i == len(seeds):
                print(f'[{i}/{len(seeds)}] 已写 {done} 条')
    print(f'完成 -> {args.out}（{done} 条）')


if __name__ == '__main__':
    main()
