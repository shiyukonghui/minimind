"""
多种向量化表示方法 - 评测与语义分析脚本
包含三部分功能：
1. 基准评测：C-Eval + CMMLU（ABCD token概率对比法）
2. 语义分析：最邻近字分析、t-SNE可视化、参数效率比
3. 对比汇总：四组方案的对比报告和可视化

用法：
  # 基准评测单个模型
  python eval_embedding_exp.py benchmark --model_path ./MiniMind2-Pretrain-512 --embed_type A

  # 语义分析单个模型
  python eval_embedding_exp.py semantic --model_path ./model/b_hf --embed_type B

  # 四组方案完整对比
  python eval_embedding_exp.py compare \
    --baseline_path ./MiniMind2-Pretrain-512 \
    --model_dir ./model \
    --loss_dir ./model
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ============================================================
#  基准评测（C-Eval + CMMLU）
# ============================================================

def load_model(model_path, device='cuda'):
    """加载 HuggingFace 格式的模型和分词器"""
    print(f'正在加载模型: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'模型参数量: {params:.2f}M')
    return model, tokenizer


def get_choice_tokens(tokenizer):
    """获取 A/B/C/D 对应的 token id"""
    choice_tokens = {}
    for letter in ['A', 'B', 'C', 'D']:
        token_id = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_id) == 1:
            choice_tokens[letter] = token_id[0]
        else:
            token_id_with_space = tokenizer.encode(f' {letter}', add_special_tokens=False)
            if len(token_id_with_space) == 1:
                choice_tokens[letter] = token_id_with_space[0]
            else:
                choice_tokens[letter] = token_id[0]
                print(f'警告: 字母 {letter} 被编码为多个 token: {token_id}, 使用第一个')
    print(f'ABCD token 映射: {choice_tokens}')
    return choice_tokens


def evaluate_multiple_choice(model, tokenizer, questions, choice_tokens, device='cuda', batch_size=8):
    """评测多选题"""
    correct = 0
    total = 0
    results_by_task = defaultdict(lambda: {'correct': 0, 'total': 0})

    prompts = []
    answers = []
    tasks = []

    for q in questions:
        prompt = f"{q['question']}\nA. {q.get('A', '')}\nB. {q.get('B', '')}\nC. {q.get('C', '')}\nD. {q.get('D', '')}\n答案是："
        prompts.append(prompt)
        answers.append(q['answer'])
        tasks.append(q.get('task', 'unknown'))

    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc='评测进度'):
        batch_prompts = prompts[i:i + batch_size]
        batch_answers = answers[i:i + batch_size]
        batch_tasks = tasks[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        last_token_logits = logits[:, -1, :]
        choice_logits = torch.stack([
            last_token_logits[:, choice_tokens[letter]] for letter in ['A', 'B', 'C', 'D']
        ], dim=-1)
        probs = torch.softmax(choice_logits, dim=-1)
        pred_indices = probs.argmax(dim=-1)
        pred_letters = [['A', 'B', 'C', 'D'][idx] for idx in pred_indices.cpu().tolist()]

        for j, (pred, ans, task) in enumerate(zip(pred_letters, batch_answers, batch_tasks)):
            total += 1
            results_by_task[task]['total'] += 1
            if pred == ans:
                correct += 1
                results_by_task[task]['correct'] += 1
            predictions.append({'pred': pred, 'answer': ans, 'task': task, 'correct': pred == ans})

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, dict(results_by_task), predictions


def run_ceval_eval(model, tokenizer, choice_tokens, device='cuda'):
    """运行 C-Eval 评测"""
    print('\n' + '=' * 60)
    print('开始 C-Eval 评测')
    print('=' * 60)

    questions = []
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files('zacharyxxxxcr/ceval-exam', repo_type='dataset')
        val_files = [f for f in files if '/val-' in f]
        subjects = sorted(set(f.split('/')[0] for f in val_files))
        print(f'C-Eval 共 {len(subjects)} 个科目')

        for subject in subjects:
            try:
                ds = load_dataset('zacharyxxxxcr/ceval-exam', subject, split='val')
                for item in ds:
                    q = {
                        'question': item.get('question', ''),
                        'A': item.get('A', ''), 'B': item.get('B', ''),
                        'C': item.get('C', ''), 'D': item.get('D', ''),
                        'answer': item.get('answer', ''), 'task': subject
                    }
                    if q['question'] and q['answer'] in ['A', 'B', 'C', 'D']:
                        questions.append(q)
            except Exception as e:
                print(f'  跳过科目 {subject}: {e}')
    except Exception as e:
        print(f'C-Eval 数据加载失败: {e}')
        return None

    print(f'C-Eval 有效题目总数: {len(questions)}')
    if not questions:
        return None

    accuracy, correct, total, by_task, predictions = evaluate_multiple_choice(
        model, tokenizer, questions, choice_tokens, device
    )

    print(f'\nC-Eval 评测结果:')
    print(f'  总准确率: {accuracy:.2%} ({correct}/{total})')
    return {'dataset': 'C-Eval', 'accuracy': accuracy, 'correct': correct, 'total': total, 'by_task': by_task}


def run_cmmlu_eval(model, tokenizer, choice_tokens, device='cuda'):
    """运行 CMMLU 评测"""
    print('\n' + '=' * 60)
    print('开始 CMMLU 评测')
    print('=' * 60)

    try:
        ds = load_dataset('svjack/cmmlu', split='train')
        questions = []
        for item in ds:
            q = {
                'question': item['question'], 'A': item['A'],
                'B': item['B'], 'C': item['C'], 'D': item['D'],
                'answer': item['answer'], 'task': item['task']
            }
            if q['question'] and q['answer'] in ['A', 'B', 'C', 'D']:
                questions.append(q)
    except Exception as e:
        print(f'CMMLU 数据加载失败: {e}')
        return None

    print(f'CMMLU 有效题目总数: {len(questions)}')
    accuracy, correct, total, by_task, predictions = evaluate_multiple_choice(
        model, tokenizer, questions, choice_tokens, device
    )

    print(f'\nCMMLU 评测结果:')
    print(f'  总准确率: {accuracy:.2%} ({correct}/{total})')
    return {'dataset': 'CMMLU', 'accuracy': accuracy, 'correct': correct, 'total': total, 'by_task': by_task}


def run_benchmark(model_path, embed_type, device='cuda', batch_size=8, dataset='all'):
    """运行基准评测"""
    model, tokenizer = load_model(model_path, device)
    choice_tokens = get_choice_tokens(tokenizer)
    results = {}

    if dataset in ['all', 'ceval']:
        try:
            ceval_result = run_ceval_eval(model, tokenizer, choice_tokens, device)
            if ceval_result:
                results['C-Eval'] = ceval_result
        except Exception as e:
            print(f'C-Eval 评测失败: {e}')

    if dataset in ['all', 'cmmlu']:
        try:
            cmmlu_result = run_cmmlu_eval(model, tokenizer, choice_tokens, device)
            if cmmlu_result:
                results['CMMLU'] = cmmlu_result
        except Exception as e:
            print(f'CMMLU 评测失败: {e}')

    # 保存结果
    output_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
    output_path = os.path.join(output_dir, f'{embed_type.lower()}_benchmark_results.json')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'评测结果已保存至: {output_path}')

    # 释放GPU显存
    del model
    torch.cuda.empty_cache()

    return results


# ============================================================
#  语义分析
# ============================================================

def get_embedding_weights(model_path, embed_type, device='cpu'):
    """从模型中提取嵌入权重

    Args:
        model_path: HuggingFace格式模型路径
        embed_type: 嵌入方案类型
        device: 计算设备

    Returns:
        mapped_weights: (vocab_size, hidden_size) 映射后的嵌入表示
        raw_weights: (vocab_size, coord_dim) 原始低维嵌入表示
    """
    # 对于方案A（标准Llama格式），直接读取embed_tokens
    if embed_type == 'A':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
        )
        mapped_weights = model.model.embed_tokens.weight.data.cpu().float()
        raw_weights = mapped_weights  # A方案原始就是高维
        del model
        torch.cuda.empty_cache()
        return mapped_weights, raw_weights

    # 对于B/C1/C2，需要使用我们的自定义模型
    from model_embedding import MiniMindEmbeddingConfig, MiniMindEmbeddingForCausalLM

    # 读取config来确定模型参数
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        hidden_size = config_data.get('hidden_size', 512)
        num_hidden_layers = config_data.get('num_hidden_layers', 8)
    else:
        hidden_size, num_hidden_layers = 512, 8

    lm_config = MiniMindEmbeddingConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        embed_type=embed_type
    )
    model = MiniMindEmbeddingForCausalLM(lm_config)

    # 加载权重
    weight_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') or f.endswith('.bin')]
    if weight_files:
        from safetensors.torch import load_file
        for wf in weight_files:
            if wf.endswith('.safetensors'):
                state_dict = load_file(os.path.join(model_path, wf))
            else:
                state_dict = torch.load(os.path.join(model_path, wf), map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=False)

    model.eval()
    mapped_weights = model.model.get_embedding_weights().cpu().float()
    raw_weights = model.model.get_raw_embedding_weights().cpu().float()

    del model
    torch.cuda.empty_cache()
    return mapped_weights, raw_weights


def analyze_nearest_neighbors(weights, tokenizer, top_k=200, show_top=20, neighbors_k=5):
    """高频字最邻近分析

    Args:
        weights: (vocab_size, dim) 嵌入权重
        tokenizer: 分词器
        top_k: 考虑前多少个高频token
        show_top: 打印前多少个token的邻居
        neighbors_k: 每个字显示几个邻居

    Returns:
        neighbors_dict: {token_id: [(neighbor_id, similarity), ...]}
    """
    import torch.nn.functional as F

    # 归一化计算余弦相似度
    normed = F.normalize(weights, dim=-1)
    sim_matrix = normed @ normed.T  # (vocab, vocab)

    # 取前top_k个token（按token_id顺序取，实际高频字靠前）
    freq_token_ids = list(range(min(top_k, weights.shape[0])))

    neighbors_dict = {}
    print(f'\n--- 最邻近字分析 (top-{show_top}/{top_k}) ---')
    for i, token_id in enumerate(freq_token_ids[:show_top]):
        sims = sim_matrix[token_id]
        top_vals = sims.topk(neighbors_k + 1)
        neighbor_ids = top_vals.indices[1:].tolist()  # 排除自身
        neighbor_sims = top_vals.values[1:].tolist()

        token_str = tokenizer.decode([token_id])
        neighbors_str = [tokenizer.decode([nid]) for nid in neighbor_ids]

        neighbors_dict[token_id] = list(zip(neighbor_ids, neighbor_sims))

        # 尝试打印中文
        try:
            neighbor_display = ', '.join(
                [f'"{n}"({s:.3f})' for n, s in zip(neighbors_str, neighbor_sims)]
            )
            print(f'  Token {token_id} ("{token_str}"): {neighbor_display}')
        except UnicodeEncodeError:
            neighbor_display = ', '.join(
                [f'id={nid}({s:.3f})' for nid, s in zip(neighbor_ids, neighbor_sims)]
            )
            print(f'  Token {token_id}: {neighbor_display}')

    return neighbors_dict


def visualize_tsne(weights, tokenizer, token_ids=None, output_path=None, title='t-SNE Visualization'):
    """t-SNE 语义拓扑可视化

    Args:
        weights: (vocab_size, dim) 嵌入权重
        tokenizer: 分词器
        token_ids: 要可视化的token id列表，默认取前200
        output_path: 图片保存路径
        title: 图表标题
    """
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if token_ids is None:
        token_ids = list(range(min(200, weights.shape[0])))

    selected_weights = weights[token_ids].numpy()

    # t-SNE降维
    if selected_weights.shape[1] <= 2:
        # 如果已经是2维，直接使用
        coords_2d = selected_weights[:, :2]
    elif selected_weights.shape[1] == 3:
        # 3维取前2维或做t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(token_ids) - 1))
        coords_2d = tsne.fit_transform(selected_weights)
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(token_ids) - 1))
        coords_2d = tsne.fit_transform(selected_weights)

    # 绘制散点图
    plt.figure(figsize=(16, 12))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.6, s=30)

    # 标注汉字
    for i, token_id in enumerate(token_ids):
        try:
            label = tokenizer.decode([token_id]).strip()
            if label and len(label) <= 4:  # 只标注短文本
                plt.annotate(label, (coords_2d[i, 0], coords_2d[i, 1]),
                             fontsize=7, alpha=0.8)
        except Exception:
            pass

    plt.title(title, fontsize=14)
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f't-SNE图已保存至: {output_path}')
    plt.close()


def compute_param_efficiency(ppl, embed_param_count, embed_type):
    """计算参数效率比

    语义耦合度 = PPL / 嵌入相关参数总量
    值越低说明参数效率越高
    """
    efficiency = ppl / embed_param_count if embed_param_count > 0 else float('inf')
    return efficiency


# 嵌入方案的参数量（预设值）
EMBED_PARAM_COUNTS = {
    'A': 6400 * 512,           # 3,276,800 (仅embed_tokens，权重绑定)
    'B': 2 * 512,              # 1,024 (仅coord_proj) + 512*6400(lm_head独立) = 3,277,824
    'C1': 3 * 512,             # 1,536 (仅coord_proj) + 512*6400(lm_head独立) = 3,278,336
    'C2': 6400 * 3 + 3 * 512,  # 20,736 (tiny_embed+tiny_proj) + 512*6400(lm_head独立) = 3,297,536
}

# 包含lm_head独立参数的完整嵌入参数量
EMBED_PARAM_COUNTS_WITH_LM_HEAD = {
    'A': 6400 * 512,              # 3,276,800 (权重绑定，lm_head共享)
    'B': 2 * 512 + 512 * 6400,    # 3,277,824
    'C1': 3 * 512 + 512 * 6400,   # 3,278,336
    'C2': 6400 * 3 + 3 * 512 + 512 * 6400,  # 3,297,536
}


def run_semantic_analysis(model_path, embed_type, device='cpu', output_dir=None):
    """运行语义分析"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path) or '.', 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'语义分析 - 方案 {embed_type}')
    print(f'{"="*60}')

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 提取嵌入权重
    mapped_weights, raw_weights = get_embedding_weights(model_path, embed_type, device)
    print(f'映射后嵌入维度: {mapped_weights.shape}')
    print(f'原始嵌入维度: {raw_weights.shape}')

    results = {'embed_type': embed_type}

    # 1. 最邻近分析
    neighbors_dict = analyze_nearest_neighbors(mapped_weights, tokenizer, top_k=200, show_top=20)
    neighbors_serializable = {}
    for tid, neighbors in neighbors_dict.items():
        neighbors_serializable[str(tid)] = [
            {'neighbor_id': int(nid), 'similarity': float(sim)} for nid, sim in neighbors
        ]
    results['nearest_neighbors'] = neighbors_serializable

    # 2. t-SNE可视化
    tsne_path = os.path.join(output_dir, f'{embed_type.lower()}_tsne.png')
    visualize_tsne(
        mapped_weights, tokenizer,
        output_path=tsne_path,
        title=f'Embedding {embed_type} - t-SNE'
    )

    # 3. 低维原始嵌入可视化（B/C1/C2）
    if embed_type != 'A' and raw_weights.shape[1] <= 3:
        raw_tsne_path = os.path.join(output_dir, f'{embed_type.lower()}_raw_embed.png')
        visualize_tsne(
            raw_weights, tokenizer,
            output_path=raw_tsne_path,
            title=f'Embedding {embed_type} - Raw Low-Dim Space'
        )

    # 4. 参数效率比（PPL需要从benchmark结果中获取）
    results['embed_param_count'] = EMBED_PARAM_COUNTS.get(embed_type, 0)
    results['embed_param_count_with_lm_head'] = EMBED_PARAM_COUNTS_WITH_LM_HEAD.get(embed_type, 0)

    # 保存分析结果
    analysis_path = os.path.join(output_dir, f'{embed_type.lower()}_semantic_analysis.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'语义分析结果已保存至: {analysis_path}')

    return results


# ============================================================
#  四组方案对比汇总
# ============================================================

def plot_loss_comparison(loss_dir, output_dir):
    """绘制四组方案的训练loss曲线对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for embed_type in ['A', 'B', 'C1', 'C2']:
        loss_path = os.path.join(loss_dir, f'{embed_type.lower()}_loss_history.json')
        if not os.path.exists(loss_path):
            continue
        with open(loss_path, 'r') as f:
            losses = json.load(f)

        # 平滑处理（移动平均）
        window = min(50, max(1, len(losses) // 20))
        if len(losses) > window:
            smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        else:
            smoothed = losses

        label = {'A': 'A: 高维基线', 'B': 'B: 二维固定坐标',
                 'C1': 'C1: 三维固定模式', 'C2': 'C2: 三维可学习瓶颈'}
        plt.plot(smoothed, label=label.get(embed_type, embed_type), alpha=0.8)

    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'loss_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Loss对比图已保存至: {output_path}')
    plt.close()


def generate_comparison_report(model_dir, loss_dir, output_dir):
    """生成四组方案的完整对比报告"""
    os.makedirs(output_dir, exist_ok=True)

    report = {
        'experiment_name': '多种向量化表示方法对比',
        'schemes': {}
    }

    # 收集各组评测和语义分析结果
    for embed_type in ['A', 'B', 'C1', 'C2']:
        scheme_data = {
            'embed_type': embed_type,
            'embed_param_count': EMBED_PARAM_COUNTS.get(embed_type, 0),
            'embed_param_count_with_lm_head': EMBED_PARAM_COUNTS_WITH_LM_HEAD.get(embed_type, 0),
        }

        # 基准评测结果
        if embed_type == 'A':
            bench_path = os.path.join(model_dir, 'a_benchmark_results.json')
        else:
            bench_path = os.path.join(model_dir, f'{embed_type.lower()}_hf', f'{embed_type.lower()}_benchmark_results.json')
            if not os.path.exists(bench_path):
                bench_path = os.path.join(model_dir, f'{embed_type.lower()}_benchmark_results.json')

        if os.path.exists(bench_path):
            with open(bench_path, 'r') as f:
                bench_data = json.load(f)
                scheme_data['benchmark'] = bench_data
                # 提取PPL（用准确率近似）
                for dataset_name, dataset_result in bench_data.items():
                    if isinstance(dataset_result, dict) and 'accuracy' in dataset_result:
                        scheme_data[f'{dataset_name}_accuracy'] = dataset_result['accuracy']

        # 语义分析结果
        analysis_path = os.path.join(model_dir, 'analysis', f'{embed_type.lower()}_semantic_analysis.json')
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                scheme_data['semantic_analysis'] = json.load(f)

        # Loss历史
        loss_path = os.path.join(loss_dir, f'{embed_type.lower()}_loss_history.json')
        if os.path.exists(loss_path):
            with open(loss_path, 'r') as f:
                losses = json.load(f)
                scheme_data['final_loss'] = losses[-1] if losses else None
                scheme_data['min_loss'] = min(losses) if losses else None

        report['schemes'][embed_type] = scheme_data

    # 参数效率比计算
    print(f'\n{"="*60}')
    print('四组方案对比汇总')
    print(f'{"="*60}')
    print(f'{"方案":<6} {"嵌入参数量":>14} {"含lm_head参数":>14} {"最终Loss":>10} {"C-Eval":>8} {"CMMLU":>8}')
    print('-' * 70)

    for embed_type in ['A', 'B', 'C1', 'C2']:
        s = report['schemes'].get(embed_type, {})
        ep = s.get('embed_param_count', 0)
        ep_lm = s.get('embed_param_count_with_lm_head', 0)
        fl = s.get('final_loss', '-')
        ceval = s.get('C-Eval_accuracy', '-')
        cmmlu = s.get('CMMLU_accuracy', '-')

        fl_str = f'{fl:.4f}' if isinstance(fl, (int, float)) else fl
        ceval_str = f'{ceval:.2%}' if isinstance(ceval, (int, float)) else str(ceval)
        cmmlu_str = f'{cmmlu:.2%}' if isinstance(cmmlu, (int, float)) else str(cmmlu)

        print(f'{embed_type:<6} {ep:>14,} {ep_lm:>14,} {fl_str:>10} {ceval_str:>8} {cmmlu_str:>8}')

    # 保存对比报告
    report_path = os.path.join(output_dir, 'comparison_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n对比报告已保存至: {report_path}')

    # 绘制loss对比图
    plot_loss_comparison(loss_dir, output_dir)

    return report


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Embedding Experiment Evaluation')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # benchmark 子命令
    bench_parser = subparsers.add_parser('benchmark', help='基准评测')
    bench_parser.add_argument('--model_path', required=True, type=str, help='HuggingFace格式模型路径')
    bench_parser.add_argument('--embed_type', required=True, type=str, choices=['A', 'B', 'C1', 'C2'])
    bench_parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    bench_parser.add_argument('--batch_size', default=8, type=int)
    bench_parser.add_argument('--dataset', default='all', type=str, choices=['all', 'ceval', 'cmmlu'])

    # semantic 子命令
    sem_parser = subparsers.add_parser('semantic', help='语义分析')
    sem_parser.add_argument('--model_path', required=True, type=str, help='HuggingFace格式模型路径')
    sem_parser.add_argument('--embed_type', required=True, type=str, choices=['A', 'B', 'C1', 'C2'])
    sem_parser.add_argument('--device', default='cpu', type=str)
    sem_parser.add_argument('--output_dir', type=str, default=None)

    # compare 子命令
    cmp_parser = subparsers.add_parser('compare', help='四组方案完整对比')
    cmp_parser.add_argument('--baseline_path', default='./MiniMind2-Pretrain-512', type=str,
                            help='方案A基线模型路径')
    cmp_parser.add_argument('--model_dir', default='./model', type=str,
                            help='B/C1/C2的HuggingFace模型所在目录')
    cmp_parser.add_argument('--loss_dir', default='./model', type=str,
                            help='Loss历史文件所在目录')
    cmp_parser.add_argument('--output_dir', default='./comparison_results', type=str,
                            help='对比报告输出目录')
    cmp_parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    cmp_parser.add_argument('--batch_size', default=8, type=int)
    cmp_parser.add_argument('--dataset', default='all', type=str, choices=['all', 'ceval', 'cmmlu'])

    args = parser.parse_args()

    if args.command == 'benchmark':
        run_benchmark(args.model_path, args.embed_type, args.device, args.batch_size, args.dataset)

    elif args.command == 'semantic':
        run_semantic_analysis(args.model_path, args.embed_type, args.device, args.output_dir)

    elif args.command == 'compare':
        # 四组方案完整对比流程
        print('=' * 60)
        print('  四组方案完整对比评测')
        print('=' * 60)

        # 1. 方案A评测
        print('\n>>> 评测方案A（基线）')
        if os.path.exists(args.baseline_path):
            run_benchmark(args.baseline_path, 'A', args.device, args.batch_size, args.dataset)
            run_semantic_analysis(args.baseline_path, 'A', args.device,
                                  os.path.join(args.output_dir, 'analysis'))
        else:
            print(f'方案A基线模型不存在: {args.baseline_path}')

        # 2. 方案B/C1/C2评测
        for embed_type in ['B', 'C1', 'C2']:
            hf_path = os.path.join(args.model_dir, f'{embed_type.lower()}_hf')
            print(f'\n>>> 评测方案{embed_type}')
            if os.path.exists(hf_path):
                run_benchmark(hf_path, embed_type, args.device, args.batch_size, args.dataset)
                run_semantic_analysis(hf_path, embed_type, args.device,
                                      os.path.join(args.output_dir, 'analysis'))
            else:
                print(f'方案{embed_type}模型不存在: {hf_path}')

        # 3. 生成对比报告
        print('\n>>> 生成对比报告')
        generate_comparison_report(args.model_dir, args.loss_dir, args.output_dir)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
