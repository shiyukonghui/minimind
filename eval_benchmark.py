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
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# STEM 学科列表（用于分类汇总）
STEM_TASKS = [
    'advanced_mathematics', 'high_school_mathematics', 'college_mathematics',
    'middle_school_mathematics', 'elementary_mathematics',
    'high_school_physics', 'college_physics', 'middle_school_physics',
    'conceptual_physics',
    'high_school_chemistry', 'college_chemistry', 'middle_school_chemistry',
    'high_school_biology', 'college_biology', 'middle_school_biology',
    'computer_science', 'computer_network', 'computer_architecture',
    'computer_security', 'operating_system', 'college_programming',
    'probability_and_statistics', 'discrete_mathematics', 'linear_algebra',
    'college_mathematics', 'machine_learning',
    'metrology_engineer', 'electrical_engineer', 'electrical_engineering',
    'basic_medicine', 'clinical_medicine', 'veterinary_medicine',
    'college_medical_statistics', 'clinical_knowledge', 'professional_medicine',
    'physics', 'mathematics', 'chemistry', 'biology',
    'agronomy', 'plant_protection', 'anatomy', 'genetics', 'virology',
    'nutrition', 'astronomy',
]

# 人文社科学科列表
HUMANITIES_TASKS = [
    'chinese_language_and_literature', 'high_school_chinese', 'elementary_chinese',
    'middle_school_history', 'high_school_history', 'modern_chinese_history',
    'chinese_history', 'world_history',
    'ideological_and_moral_cultivation', 'legal_and_moral_basis',
    'marxism', 'mao_zedong_thought', 'marxist_theory',
    'law', 'legal_professional', 'college_law', 'professional_law',
    'international_law', 'jurisprudence',
    'civil_servant', 'chinese_civil_service_exam', 'teacher_qualification',
    'chinese_teacher_qualification',
    'sports_science', 'art_studies', 'arts', 'professional_tour_guide',
    'history', 'politics', 'philosophy', 'high_school_politics',
    'middle_school_politics',
    'chinese_foreign_policy', 'security_study', 'sociology',
    'journalism', 'public_relations', 'business_ethics', 'management',
    'marketing', 'economics', 'college_economics',
    'education', 'education_science', 'college_education',
    'ancient_chinese', 'modern_chinese', 'chinese_literature',
    'chinese_food_culture', 'traditional_chinese_medicine',
    'world_religions', 'ethnology', 'professional_psychology',
    'human_sexuality',
]

# 英文科目名 -> 中文名映射
TASK_NAME_ZH = {
    'accountant': '会计', 'advanced_mathematics': '高等数学',
    'art_studies': '艺术学', 'basic_medicine': '基础医学',
    'business_administration': '工商管理', 'chinese_language_and_literature': '中国语言文学',
    'civil_servant': '公务员', 'clinical_medicine': '临床医学',
    'college_chemistry': '大学化学', 'college_economics': '大学经济学',
    'college_physics': '大学物理', 'college_programming': '大学编程',
    'computer_architecture': '计算机架构', 'computer_network': '计算机网络',
    'discrete_mathematics': '离散数学', 'education_science': '教育学',
    'electrical_engineer': '电气工程师', 'environmental_impact_assessment_engineer': '环境影响评估工程师',
    'fire_engineer': '消防工程师', 'high_school_biology': '高中生物',
    'high_school_chemistry': '高中化学', 'high_school_chinese': '高中语文',
    'high_school_geography': '高中地理', 'high_school_history': '高中历史',
    'high_school_mathematics': '高中数学', 'high_school_physics': '高中物理',
    'high_school_politics': '高中政治', 'ideological_and_moral_cultivation': '思想道德修养',
    'law': '法律', 'legal_professional': '法律职业', 'logic': '逻辑学',
    'mao_zedong_thought': '毛泽东思想', 'marxism': '马克思主义',
    'metrology_engineer': '计量工程师', 'middle_school_biology': '中学生物',
    'middle_school_chemistry': '中学化学', 'middle_school_geography': '中学地理',
    'middle_school_history': '中学历史', 'middle_school_mathematics': '中学数学',
    'middle_school_physics': '中学物理', 'middle_school_politics': '中学政治',
    'modern_chinese_history': '近代中国史', 'operating_system': '操作系统',
    'physician': '医师', 'plant_protection': '植物保护',
    'probability_and_statistics': '概率与统计', 'professional_tour_guide': '导游',
    'sports_science': '体育科学', 'tax_accountant': '税务师',
    'teacher_qualification': '教师资格', 'urban_and_rural_planner': '城乡规划师',
    'veterinary_medicine': '兽医学',
    # CMMLU 科目
    'agronomy': '农学', 'anatomy': '解剖学', 'ancient_chinese': '古汉语',
    'arts': '艺术', 'astronomy': '天文学', 'business_ethics': '商业伦理',
    'chinese_civil_service_exam': '公务员考试', 'chinese_driving_rule': '中国交通规则',
    'chinese_food_culture': '中国饮食文化', 'chinese_foreign_policy': '中国外交政策',
    'chinese_history': '中国历史', 'chinese_literature': '中国文学',
    'chinese_teacher_qualification': '教师资格考试', 'clinical_knowledge': '临床知识',
    'college_actuarial_science': '大学精算学', 'college_education': '大学教育',
    'college_engineering_hydrology': '大学工程水文学', 'college_law': '大学法律',
    'college_mathematics': '大学数学', 'college_medical_statistics': '医学统计学',
    'college_medicine': '大学医学', 'computer_science': '计算机科学',
    'computer_security': '计算机安全', 'conceptual_physics': '概念物理',
    'construction_project_management': '建设工程管理', 'economics': '经济学',
    'education': '教育学', 'electrical_engineering': '电气工程',
    'elementary_chinese': '小学语文', 'elementary_commonsense': '小学常识',
    'elementary_information_and_technology': '小学信息技术', 'elementary_mathematics': '小学数学',
    'ethnology': '民族学', 'food_science': '食品科学', 'genetics': '遗传学',
    'global_facts': '全球事实', 'high_school_geography': '高中地理',
    'human_sexuality': '人类性学', 'international_law': '国际法',
    'journalism': '新闻学', 'jurisprudence': '法理学',
    'legal_and_moral_basis': '法律与道德基础', 'logical': '逻辑学',
    'machine_learning': '机器学习', 'management': '管理学', 'marketing': '市场营销',
    'marxist_theory': '马克思主义理论', 'modern_chinese': '现代汉语',
    'nutrition': '营养学', 'philosophy': '哲学',
    'professional_accounting': '专业会计', 'professional_law': '专业法律',
    'professional_medicine': '专业医学', 'professional_psychology': '专业心理学',
    'public_relations': '公共关系', 'security_study': '安全研究',
    'sociology': '社会学', 'traditional_chinese_medicine': '中医学',
    'virology': '病毒学', 'world_history': '世界历史', 'world_religions': '世界宗教',
}


def load_model(model_path, device='cuda'):
    """加载 HuggingFace 格式的模型和分词器"""
    print(f'正在加载模型: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
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


def compute_category_stats(result, category_name, category_tasks):
    """计算某个学科大类的统计数据"""
    cat_correct = 0
    cat_total = 0
    for task in category_tasks:
        if task in result['by_task']:
            cat_correct += result['by_task'][task]['correct']
            cat_total += result['by_task'][task]['total']
    if cat_total > 0:
        return {'accuracy': cat_correct / cat_total, 'correct': cat_correct, 'total': cat_total}
    return None


def infer_model_label(model_path):
    """从模型路径推断模型标签，用于报告文件名前缀"""
    model_name = os.path.basename(model_path.rstrip('/\\')).lower()
    if 'dst' in model_name:
        return 'dst'
    elif 'pretrain' in model_name:
        return 'baseline'
    else:
        return os.path.basename(model_path.rstrip('/\\'))


def generate_report(results, model_path, model_params_m=None, model_label=None, output_dir=None):
    """生成 Markdown 格式的评测报告并保存到 reports/ 目录"""
    date_str = datetime.now().strftime('%Y%m%d')
    if output_dir is None:
        output_dir = os.path.join('.', 'reports', date_str)
    os.makedirs(output_dir, exist_ok=True)

    # 推断模型标签（用于文件名前缀，避免同日评测不同模型时文件覆盖）
    if model_label is None:
        model_label = infer_model_label(model_path)

    # 保存 JSON 结果（文件名加模型标签前缀）
    json_path = os.path.join(output_dir, f'{model_label}_benchmark_results.json')
    serializable_results = {}
    for name, result in results.items():
        r = dict(result)
        r['by_task'] = {k: dict(v) for k, v in r['by_task'].items()}
        serializable_results[name] = r
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    # 提取模型名称
    model_name = os.path.basename(model_path.rstrip('/\\'))
    date_display = datetime.now().strftime('%Y-%m-%d')

    # 构建 Markdown 报告
    lines = []
    lines.append(f'# {model_name} 模型性能评测报告')
    lines.append('')
    lines.append(f'**评测日期**: {date_display}')
    lines.append(f'**模型路径**: `{os.path.abspath(model_path)}`')
    if model_params_m is not None:
        lines.append(f'**模型参数**: {model_params_m:.2f}M')
    lines.append(f'**评测方法**: ABCD token 概率对比法（与 lm-evaluation-harness 一致）')
    lines.append('')
    lines.append('---')
    lines.append('')

    # 评测结果总览表
    lines.append('## 评测结果总览')
    lines.append('')
    lines.append('| 榜单 | 总准确率 | 正确/总题数 | STEM | 人文社科 |')
    lines.append('|------|---------|-----------|------|---------|')
    for name, result in results.items():
        stem = compute_category_stats(result, 'STEM', STEM_TASKS)
        hum = compute_category_stats(result, '人文社科', HUMANITIES_TASKS)
        stem_str = f'{stem["accuracy"]:.2%}' if stem else '-'
        hum_str = f'{hum["accuracy"]:.2%}' if hum else '-'
        lines.append(f'| **{name}** | {result["accuracy"]:.2%} | {result["correct"]}/{result["total"]} | {stem_str} | {hum_str} |')
    lines.append('')
    lines.append('> 四选一随机基线为 25%，模型整体表现接近随机猜测水平。')
    lines.append('')
    lines.append('---')
    lines.append('')

    # 各数据集的科目明细
    for name, result in results.items():
        by_task = result['by_task']
        num_subjects = len(by_task)
        total_q = result['total']
        lines.append(f'## {name} 各科目成绩（{num_subjects} 科目，{total_q} 题）')
        lines.append('')
        lines.append('| 科目 | 准确率 | 正确/总数 |')
        lines.append('|------|--------|----------|')
        # 按准确率降序排列
        sorted_tasks = sorted(
            by_task.items(),
            key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0,
            reverse=True
        )
        for task, stats in sorted_tasks:
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            zh_name = TASK_NAME_ZH.get(task, task)
            lines.append(f'| {zh_name} | {acc:.2%} | {stats["correct"]}/{stats["total"]} |')
        lines.append('')
        lines.append('---')
        lines.append('')

    # 评测环境
    lines.append('## 评测环境')
    lines.append('')
    lines.append(f'- **Python**: {__import__("sys").version.split()[0]}')
    lines.append(f'- **PyTorch**: {torch.__version__}')
    lines.append(f'- **GPU**: {"CUDA 可用" if torch.cuda.is_available() else "CUDA 不可用"}')
    lines.append(f'- **数据集来源**: C-Eval (zacharyxxxxcr/ceval-exam, val split), CMMLU (svjack/cmmlu, train split)')
    lines.append(f'- **HuggingFace 镜像**: https://hf-mirror.com')
    lines.append('')

    # 保存 Markdown 报告（文件名加模型标签前缀）
    md_path = os.path.join(output_dir, f'{model_label}_eval_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_dir, md_path, json_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MiniMind 模型性能评分测试')
    parser.add_argument('--model_path', default='./MiniMind2', type=str, help='HuggingFace 格式模型路径')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--batch_size', default=8, type=int, help='评测批大小')
    parser.add_argument('--dataset', default='all', type=str, choices=['all', 'ceval', 'cmmlu'], help='评测数据集选择')
    parser.add_argument('--output_dir', default=None, type=str, help='报告输出目录（默认 reports/YYYYMMDD）')
    parser.add_argument('--model_label', default=None, type=str, help='模型标签（用于报告文件名前缀，默认从模型路径自动推断）')
    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(args.model_path, args.device)
    model_params_m = sum(p.numel() for p in model.parameters()) / 1e6
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

    # 汇总报告（控制台输出）
    print('\n' + '=' * 60)
    print('评测汇总报告')
    print('=' * 60)
    for name, result in results.items():
        print(f'\n{name}:')
        print(f'  总准确率: {result["accuracy"]:.2%} ({result["correct"]}/{result["total"]})')
        stem = compute_category_stats(result, 'STEM', STEM_TASKS)
        hum = compute_category_stats(result, '人文社科', HUMANITIES_TASKS)
        if stem:
            print(f'  STEM 准确率: {stem["accuracy"]:.2%} ({stem["correct"]}/{stem["total"]})')
        if hum:
            print(f'  人文社科 准确率: {hum["accuracy"]:.2%} ({hum["correct"]}/{hum["total"]})')

    # 生成报告（JSON + Markdown）
    output_dir, md_path, json_path = generate_report(
        results, args.model_path, model_params_m, args.model_label, args.output_dir
    )
    print(f'\n评测结果已保存至: {json_path}')
    print(f'可读报告已保存至: {md_path}')


if __name__ == '__main__':
    main()
