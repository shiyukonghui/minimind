"""
MiniMind 模型性能评分测试脚本
使用 ABCD token 概率对比法进行多选题评测（与 lm_eval 方法一致）
评测数据集：C-Eval (validation) + CMMLU (test)
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def load_model(model_path, device='cuda'):
    """加载 HuggingFace 格式的模型和分词器"""
    print(f'正在加载模型: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
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
        # 取第一个 token（某些分词器可能将单个字母编码为多个 token）
        if len(token_id) == 1:
            choice_tokens[letter] = token_id[0]
        else:
            # 如果字母被编码为多个 token，尝试加空格
            token_id_with_space = tokenizer.encode(f' {letter}', add_special_tokens=False)
            if len(token_id_with_space) == 1:
                choice_tokens[letter] = token_id_with_space[0]
            else:
                # 使用第一个 token 作为近似
                choice_tokens[letter] = token_id[0]
                print(f'警告: 字母 {letter} 被编码为多个 token: {token_id}, 使用第一个')
    print(f'ABCD token 映射: {choice_tokens}')
    return choice_tokens


def evaluate_multiple_choice(model, tokenizer, questions, choice_tokens, device='cuda', batch_size=8):
    """
    评测多选题
    方法：对于每道题，将题目和选项构建为 prompt，获取答案位置处 A/B/C/D 四个 token 的概率，
    选择概率最高的作为模型预测，与正确答案比较计算准确率。
    """
    correct = 0
    total = 0
    results_by_task = defaultdict(lambda: {'correct': 0, 'total': 0})

    # 构建所有 prompt
    prompts = []
    answers = []
    tasks = []

    for q in questions:
        question_text = q['question']
        option_a = q.get('A', '')
        option_b = q.get('B', '')
        option_c = q.get('C', '')
        option_d = q.get('D', '')
        answer = q['answer']
        task = q.get('task', 'unknown')

        prompt = f"{question_text}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n答案是："

        prompts.append(prompt)
        answers.append(answer)
        tasks.append(task)

    # 分批推理
    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc='评测进度'):
        batch_prompts = prompts[i:i + batch_size]
        batch_answers = answers[i:i + batch_size]
        batch_tasks = tasks[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 获取最后一个 token 位置的 logits
        last_token_logits = logits[:, -1, :]

        # 提取 A/B/C/D 对应 token 的 logits
        choice_logits = torch.stack([
            last_token_logits[:, choice_tokens[letter]] for letter in ['A', 'B', 'C', 'D']
        ], dim=-1)

        # softmax 得到概率
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
    return accuracy, correct, total, results_by_task, predictions


def run_ceval_eval(model, tokenizer, choice_tokens, device='cuda'):
    """运行 C-Eval 评测"""
    print('\n' + '=' * 60)
    print('开始 C-Eval 评测')
    print('=' * 60)

    # 加载 C-Eval validation 数据
    questions = []
    # 获取 C-Eval 数据集的所有子任务
    from huggingface_hub import list_repo_files
    files = list_repo_files('zacharyxxxxcr/ceval-exam', repo_type='dataset')
    # 提取所有有 val 集的科目
    val_files = [f for f in files if '/val-' in f]
    subjects = sorted(set(f.split('/')[0] for f in val_files))
    print(f'C-Eval 共 {len(subjects)} 个科目: {subjects}')

    for subject in subjects:
        try:
            ds = load_dataset('zacharyxxxxcr/ceval-exam', subject, split='val')
            for item in ds:
                q = {
                    'question': item.get('question', ''),
                    'A': item.get('A', ''),
                    'B': item.get('B', ''),
                    'C': item.get('C', ''),
                    'D': item.get('D', ''),
                    'answer': item.get('answer', ''),
                    'task': subject
                }
                if q['question'] and q['answer'] in ['A', 'B', 'C', 'D']:
                    questions.append(q)
        except Exception as e:
            print(f'  跳过科目 {subject}: {e}')

    print(f'C-Eval 有效题目总数: {len(questions)}')

    if not questions:
        print('C-Eval: 无有效题目，跳过评测')
        return None

    accuracy, correct, total, by_task, predictions = evaluate_multiple_choice(
        model, tokenizer, questions, choice_tokens, device
    )

    print(f'\nC-Eval 评测结果:')
    print(f'  总准确率: {accuracy:.2%} ({correct}/{total})')
    print(f'\n  各科目准确率:')
    for task in sorted(by_task.keys()):
        t = by_task[task]
        acc = t['correct'] / t['total'] if t['total'] > 0 else 0
        print(f'    {task}: {acc:.2%} ({t["correct"]}/{t["total"]})')

    return {
        'dataset': 'C-Eval',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'by_task': dict(by_task)
    }


def run_cmmlu_eval(model, tokenizer, choice_tokens, device='cuda'):
    """运行 CMMLU 评测"""
    print('\n' + '=' * 60)
    print('开始 CMMLU 评测')
    print('=' * 60)

    ds = load_dataset('svjack/cmmlu', split='train')
    questions = []
    for item in ds:
        q = {
            'question': item['question'],
            'A': item['A'],
            'B': item['B'],
            'C': item['C'],
            'D': item['D'],
            'answer': item['answer'],
            'task': item['task']
        }
        if q['question'] and q['answer'] in ['A', 'B', 'C', 'D']:
            questions.append(q)

    print(f'CMMLU 有效题目总数: {len(questions)}')

    accuracy, correct, total, by_task, predictions = evaluate_multiple_choice(
        model, tokenizer, questions, choice_tokens, device
    )

    print(f'\nCMMLU 评测结果:')
    print(f'  总准确率: {accuracy:.2%} ({correct}/{total})')
    print(f'\n  各科目准确率:')
    for task in sorted(by_task.keys()):
        t = by_task[task]
        acc = t['correct'] / t['total'] if t['total'] > 0 else 0
        print(f'    {task}: {acc:.2%} ({t["correct"]}/{t["total"]})')

    return {
        'dataset': 'CMMLU',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'by_task': dict(by_task)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MiniMind 模型性能评分测试')
    parser.add_argument('--model_path', default='./MiniMind2', type=str, help='HuggingFace 格式模型路径')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--batch_size', default=8, type=int, help='评测批大小')
    parser.add_argument('--dataset', default='all', type=str, choices=['all', 'ceval', 'cmmlu'], help='评测数据集选择')
    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(args.model_path, args.device)
    choice_tokens = get_choice_tokens(tokenizer)

    results = {}

    # C-Eval 评测
    if args.dataset in ['all', 'ceval']:
        try:
            ceval_result = run_ceval_eval(model, tokenizer, choice_tokens, args.device)
            if ceval_result:
                results['C-Eval'] = ceval_result
        except Exception as e:
            print(f'C-Eval 评测失败: {e}')

    # CMMLU 评测
    if args.dataset in ['all', 'cmmlu']:
        try:
            cmmlu_result = run_cmmlu_eval(model, tokenizer, choice_tokens, args.device)
            if cmmlu_result:
                results['CMMLU'] = cmmlu_result
        except Exception as e:
            print(f'CMMLU 评测失败: {e}')

    # 汇总报告
    print('\n' + '=' * 60)
    print('评测汇总报告')
    print('=' * 60)
    for name, result in results.items():
        print(f'\n{name}:')
        print(f'  总准确率: {result["accuracy"]:.2%} ({result["correct"]}/{result["total"]})')
        # 按学科大类汇总
        stem_tasks = ['advanced_mathematics', 'high_school_mathematics', 'college_mathematics',
                      'high_school_physics', 'college_physics', 'middle_school_physics',
                      'high_school_chemistry', 'college_chemistry', 'middle_school_chemistry',
                      'high_school_biology', 'college_biology', 'middle_school_biology',
                      'computer_science', 'computer_network', 'operating_system',
                      'probability_and_statistics', 'discrete_mathematics', 'linear_algebra',
                      'metrology_engineer', 'electrical_engineer', 'machine_learning',
                      'basic_medicine', 'clinical_medicine', 'veterinary_medicine',
                      'physics', 'mathematics', 'chemistry', 'biology',
                      'agronomy', 'plant_protection']
        humanities_tasks = ['chinese_language_and_literature', 'high_school_chinese',
                           'middle_school_history', 'high_school_history',
                           'modern_chinese_history', 'ideological_and_moral_cultivation',
                           'marxism', 'mao_zedong_thought', 'law', 'legal_professional',
                           'civil_servant', 'teacher_qualification', 'sports_science',
                           'art_studies', 'professional_tour_guide',
                           'history', 'politics', 'philosophy']

        for category_name, category_tasks in [('STEM', stem_tasks), ('人文社科', humanities_tasks)]:
            cat_correct = 0
            cat_total = 0
            for task in category_tasks:
                if task in result['by_task']:
                    cat_correct += result['by_task'][task]['correct']
                    cat_total += result['by_task'][task]['total']
            if cat_total > 0:
                print(f'  {category_name} 准确率: {cat_correct / cat_total:.2%} ({cat_correct}/{cat_total})')

    # 保存结果
    output_path = './eval_results/benchmark_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 将 by_task 中的 defaultdict 转为普通 dict 并序列化
    serializable_results = {}
    for name, result in results.items():
        r = dict(result)
        r['by_task'] = {k: dict(v) for k, v in r['by_task'].items()}
        serializable_results[name] = r
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    print(f'\n评测结果已保存至: {output_path}')


if __name__ == '__main__':
    main()
