"""
DST 训练效果对比评测报告生成脚本
读取基线模型和 DST 模型的评测结果 JSON，生成对比报告和雷达图
"""
import os
import json
import argparse
from datetime import datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # 无头模式，避免 GUI 依赖
    import matplotlib.pyplot as plt
except ImportError:
    print('错误: 需要安装 matplotlib，请运行: pip install matplotlib')
    raise SystemExit(1)

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

# ==================== 子领域科目映射 ====================

CEVAL_SUBFIELDS = {
    '数学': [
        'advanced_mathematics', 'high_school_mathematics', 'middle_school_mathematics',
        'discrete_mathematics', 'probability_and_statistics',
    ],
    '物理': [
        'college_physics', 'high_school_physics', 'middle_school_physics',
    ],
    '化学/生物/医学': [
        'college_chemistry', 'high_school_chemistry', 'middle_school_chemistry',
        'high_school_biology', 'middle_school_biology', 'basic_medicine',
        'clinical_medicine', 'veterinary_medicine', 'plant_protection',
    ],
    '计算机': [
        'college_programming', 'computer_architecture', 'computer_network', 'operating_system',
    ],
    '工程/应用': [
        'electrical_engineer', 'metrology_engineer', 'fire_engineer',
        'environmental_impact_assessment_engineer',
    ],
    '人文社科': [
        'law', 'legal_professional', 'marxism', 'mao_zedong_thought',
        'ideological_and_moral_cultivation', 'high_school_politics', 'middle_school_politics',
        'high_school_history', 'middle_school_history', 'modern_chinese_history',
        'education_science', 'teacher_qualification', 'civil_servant', 'accountant',
        'tax_accountant', 'business_administration', 'urban_and_rural_planner',
        'physician', 'professional_tour_guide', 'sports_science', 'art_studies',
        'chinese_language_and_literature', 'high_school_chinese', 'high_school_geography',
        'middle_school_geography', 'middle_school_chemistry', 'logic',
        'college_economics',
    ],
}

CMMLU_SUBFIELDS = {
    '数学/逻辑': [
        'college_mathematics', 'elementary_mathematics', 'college_actuarial_science',
        'logical', 'machine_learning',
    ],
    '物理/工程': [
        'conceptual_physics', 'high_school_physics', 'electrical_engineering',
        'college_engineering_hydrology', 'construction_project_management',
    ],
    '化学/生物/医学': [
        'anatomy', 'genetics', 'virology', 'nutrition', 'college_medicine',
        'professional_medicine', 'clinical_knowledge', 'traditional_chinese_medicine',
        'food_science', 'high_school_biology', 'high_school_chemistry',
        'agronomy', 'astronomy',
    ],
    '计算机': [
        'computer_science', 'computer_security',
    ],
    '人文/社科/法律': [
        'chinese_history', 'world_history', 'ancient_chinese', 'modern_chinese',
        'chinese_literature', 'philosophy', 'sociology', 'journalism',
        'public_relations', 'international_law', 'college_law', 'professional_law',
        'jurisprudence', 'legal_and_moral_basis', 'chinese_foreign_policy',
        'security_study', 'ethnology', 'world_religions',
    ],
    '经济/管理/教育': [
        'economics', 'management', 'marketing', 'business_ethics',
        'education', 'college_education', 'chinese_teacher_qualification',
        'professional_psychology', 'professional_accounting',
        'chinese_civil_service_exam', 'chinese_food_culture',
        'chinese_driving_rule', 'human_sexuality', 'elementary_chinese',
        'elementary_commonsense', 'elementary_information_and_technology',
        'college_medical_statistics', 'high_school_geography', 'high_school_mathematics',
        'high_school_politics', 'arts', 'global_facts', 'sports_science',
        'public_relations', 'marxist_theory',
    ],
}

# ==================== 中文字体配置 ====================

def setup_chinese_font():
    """配置 matplotlib 中文字体，优先 SimHei，不可用则回退"""
    font_candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    for font_name in font_candidates:
        try:
            from matplotlib.font_manager import FontProperties
            fp = FontProperties(family=font_name)
            if fp.get_name() != font_name and fp.get_name() not in font_candidates:
                continue
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return font_name
        except Exception:
            continue
    # 回退：使用默认字体，关闭 unicode_minus
    plt.rcParams['axes.unicode_minus'] = False
    return None


# ==================== 数据加载与聚合 ====================

def load_results(json_path):
    """加载 benchmark_results.json"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_category_accuracy(result, tasks):
    """计算指定科目列表的聚合准确率"""
    cat_correct = 0
    cat_total = 0
    for task in tasks:
        if task in result['by_task']:
            cat_correct += result['by_task'][task]['correct']
            cat_total += result['by_task'][task]['total']
    if cat_total > 0:
        return cat_correct / cat_total
    return None


def compute_subfield_accuracies(result, subfield_map):
    """计算子领域准确率映射，返回 {子领域名: 准确率}"""
    acc_map = {}
    for subfield_name, tasks in subfield_map.items():
        acc = compute_category_accuracy(result, tasks)
        if acc is not None:
            acc_map[subfield_name] = acc
    return acc_map


def compute_overall_dimensions(baseline_results, dst_results):
    """计算总体能力对比的 6 个维度数据"""
    dimensions = []
    baseline_values = []
    dst_values = []

    for dataset_name in ['C-Eval', 'CMMLU']:
        if dataset_name not in baseline_results or dataset_name not in dst_results:
            continue
        b = baseline_results[dataset_name]
        d = dst_results[dataset_name]

        # 总准确率
        dimensions.append(f'{dataset_name} 总体')
        baseline_values.append(b['accuracy'] * 100)
        dst_values.append(d['accuracy'] * 100)

        # STEM
        b_stem = compute_category_accuracy(b, STEM_TASKS)
        d_stem = compute_category_accuracy(d, STEM_TASKS)
        if b_stem is not None and d_stem is not None:
            dimensions.append(f'{dataset_name} STEM')
            baseline_values.append(b_stem * 100)
            dst_values.append(d_stem * 100)

        # 人文社科
        b_hum = compute_category_accuracy(b, HUMANITIES_TASKS)
        d_hum = compute_category_accuracy(d, HUMANITIES_TASKS)
        if b_hum is not None and d_hum is not None:
            dimensions.append(f'{dataset_name} 人文社科')
            baseline_values.append(b_hum * 100)
            dst_values.append(d_hum * 100)

    return dimensions, baseline_values, dst_values


def compute_task_deltas(baseline_results, dst_results, dataset_name):
    """计算两个模型在指定数据集上的各科目差异，返回排序后的 [(科目, 基线准确率, DST准确率, 差异)] 列表"""
    if dataset_name not in baseline_results or dataset_name not in dst_results:
        return []

    b_tasks = baseline_results[dataset_name]['by_task']
    d_tasks = dst_results[dataset_name]['by_task']
    common_tasks = set(b_tasks.keys()) & set(d_tasks.keys())

    deltas = []
    for task in common_tasks:
        b_acc = b_tasks[task]['correct'] / b_tasks[task]['total'] if b_tasks[task]['total'] > 0 else 0
        d_acc = d_tasks[task]['correct'] / d_tasks[task]['total'] if d_tasks[task]['total'] > 0 else 0
        deltas.append((task, b_acc, d_acc, d_acc - b_acc))

    # 按差异降序排列（最大提升在前）
    deltas.sort(key=lambda x: x[3], reverse=True)
    return deltas


# ==================== 雷达图绘制 ====================

def draw_radar_chart(labels, values_dict, title, save_path, value_range=None):
    """
    绘制雷达图
    - labels: 维度标签列表
    - values_dict: {模型名: [值列表]}，支持 2-3 个模型
    - title: 图标题
    - save_path: 保存路径
    - value_range: (min, max) 值域范围，默认自动
    """
    num_vars = len(labels)
    if num_vars < 3:
        print(f'警告: 雷达图维度不足 ({num_vars})，跳过绘制: {title}')
        return

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#4285F4', '#EA4335', '#34A853']  # 蓝、红、绿
    fill_alphas = [0.15, 0.15, 0.10]

    for idx, (model_name, values) in enumerate(values_dict.items()):
        # 闭合多边形
        values_closed = values + values[:1]
        angles_closed = angles + angles[:1]

        color = colors[idx % len(colors)]
        ax.plot(angles_closed, values_closed, 'o-', linewidth=2, label=model_name,
                color=color, markersize=6)
        ax.fill(angles_closed, values_closed, alpha=fill_alphas[idx % len(fill_alphas)],
                color=color)

    # 设置刻度标签
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)

    # 设置值域范围
    if value_range:
        ax.set_ylim(value_range[0], value_range[1])
    else:
        all_vals = [v for vals in values_dict.values() for v in vals]
        if all_vals:
            vmin = max(0, min(all_vals) - 5)
            vmax = max(all_vals) + 5
            ax.set_ylim(vmin, vmax)

    # 刻度格式
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    ax.set_title(title, size=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'雷达图已保存: {save_path}')


def draw_delta_radar(labels, baseline_values, dst_values, title, save_path):
    """绘制差异雷达图（Delta Radar），展示 DST 相对基线的百分比差异"""
    num_vars = len(labels)
    if num_vars < 3:
        print(f'警告: 雷达图维度不足 ({num_vars})，跳过绘制: {title}')
        return

    deltas = [d - b for b, d in zip(baseline_values, dst_values)]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 闭合
    deltas_closed = deltas + deltas[:1]
    angles_closed = angles + angles[:1]

    # 正值用绿色，负值用红色
    ax.plot(angles_closed, deltas_closed, 'o-', linewidth=2, label='DST - 基线',
            color='#5F6368', markersize=6)
    ax.fill(angles_closed, deltas_closed, alpha=0.2, color='#5F6368')

    # 画零线
    ax.plot(angles_closed, [0] * len(angles_closed), '--', linewidth=1, color='gray', alpha=0.6)

    # 设置刻度标签
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)

    # 自动值域
    abs_max = max(abs(min(deltas)), abs(max(deltas)), 1)
    ax.set_ylim(-abs_max - 1, abs_max + 1)

    # 刻度格式
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.1f}%'))

    ax.set_title(title, size=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'差异雷达图已保存: {save_path}')


# ==================== 对比报告生成 ====================

def generate_comparison_report(baseline_path, dst_path, baseline_label, dst_label, output_dir):
    """主函数：加载数据、计算、绘图、生成 Markdown 报告"""
    os.makedirs(output_dir, exist_ok=True)

    # 配置中文字体
    font_name = setup_chinese_font()
    if font_name:
        print(f'使用中文字体: {font_name}')
    else:
        print('未找到中文字体，雷达图标签可能显示为方块，建议安装 SimHei 字体')

    # 加载数据
    print(f'\n加载基线结果: {baseline_path}')
    baseline_results = load_results(baseline_path)
    print(f'加载 DST 结果: {dst_path}')
    dst_results = load_results(dst_path)

    # 模型信息
    baseline_model_name = os.path.basename(baseline_path).replace('_benchmark_results.json', '')
    dst_model_name = os.path.basename(dst_path).replace('_benchmark_results.json', '')

    date_display = datetime.now().strftime('%Y-%m-%d')

    # ---- 1. 总体能力对比 ----
    print('\n计算总体能力对比...')
    dimensions, baseline_overall, dst_overall = compute_overall_dimensions(baseline_results, dst_results)

    # 绘制总体能力雷达图
    radar_overall_path = os.path.join(output_dir, 'comparison_radar_overall.png')
    # 计算合理的值域范围
    all_overall_vals = baseline_overall + dst_overall
    overall_vmin = max(0, min(all_overall_vals) - 5)
    overall_vmax = max(all_overall_vals) + 5
    draw_radar_chart(
        dimensions,
        {baseline_label: baseline_overall, dst_label: dst_overall},
        '总体能力对比',
        radar_overall_path,
        value_range=(overall_vmin, overall_vmax)
    )

    # ---- 2. C-Eval 子领域对比 ----
    print('计算 C-Eval 子领域对比...')
    if 'C-Eval' in baseline_results and 'C-Eval' in dst_results:
        b_ceval_sub = compute_subfield_accuracies(baseline_results['C-Eval'], CEVAL_SUBFIELDS)
        d_ceval_sub = compute_subfield_accuracies(dst_results['C-Eval'], CEVAL_SUBFIELDS)
        ceval_sub_labels = [k for k in b_ceval_sub if k in d_ceval_sub]
        b_ceval_vals = [b_ceval_sub[k] * 100 for k in ceval_sub_labels]
        d_ceval_vals = [d_ceval_sub[k] * 100 for k in ceval_sub_labels]

        radar_ceval_path = os.path.join(output_dir, 'comparison_radar_ceval.png')
        all_ceval_vals = b_ceval_vals + d_ceval_vals
        ceval_vmin = max(0, min(all_ceval_vals) - 5)
        ceval_vmax = max(all_ceval_vals) + 5
        draw_radar_chart(
            ceval_sub_labels,
            {baseline_label: b_ceval_vals, dst_label: d_ceval_vals},
            'C-Eval 子领域对比',
            radar_ceval_path,
            value_range=(ceval_vmin, ceval_vmax)
        )
    else:
        ceval_sub_labels = b_ceval_vals = d_ceval_vals = []
        radar_ceval_path = None

    # ---- 3. CMMLU 子领域对比 ----
    print('计算 CMMLU 子领域对比...')
    if 'CMMLU' in baseline_results and 'CMMLU' in dst_results:
        b_cmmlu_sub = compute_subfield_accuracies(baseline_results['CMMLU'], CMMLU_SUBFIELDS)
        d_cmmlu_sub = compute_subfield_accuracies(dst_results['CMMLU'], CMMLU_SUBFIELDS)
        cmmlu_sub_labels = [k for k in b_cmmlu_sub if k in d_cmmlu_sub]
        b_cmmlu_vals = [b_cmmlu_sub[k] * 100 for k in cmmlu_sub_labels]
        d_cmmlu_vals = [d_cmmlu_sub[k] * 100 for k in cmmlu_sub_labels]

        radar_cmmlu_path = os.path.join(output_dir, 'comparison_radar_cmmlu.png')
        all_cmmlu_vals = b_cmmlu_vals + d_cmmlu_vals
        cmmlu_vmin = max(0, min(all_cmmlu_vals) - 5)
        cmmlu_vmax = max(all_cmmlu_vals) + 5
        draw_radar_chart(
            cmmlu_sub_labels,
            {baseline_label: b_cmmlu_vals, dst_label: d_cmmlu_vals},
            'CMMLU 子领域对比',
            radar_cmmlu_path,
            value_range=(cmmlu_vmin, cmmlu_vmax)
        )
    else:
        cmmlu_sub_labels = b_cmmlu_vals = d_cmmlu_vals = []
        radar_cmmlu_path = None

    # ---- 4. 差异雷达图 ----
    print('绘制差异雷达图...')
    radar_delta_path = os.path.join(output_dir, 'comparison_radar_delta.png')
    draw_delta_radar(
        dimensions, baseline_overall, dst_overall,
        'DST 相对基线差异（百分点）',
        radar_delta_path
    )

    # ---- 5. 科目差异排名 ----
    ceval_deltas = compute_task_deltas(baseline_results, dst_results, 'C-Eval')
    cmmlu_deltas = compute_task_deltas(baseline_results, dst_results, 'CMMLU')

    # ---- 6. 生成 Markdown 报告 ----
    print('\n生成对比报告...')
    lines = []
    lines.append('# DST 训练效果对比评测报告')
    lines.append('')
    lines.append(f'**生成日期**: {date_display}')
    lines.append(f'**对比模型**: {baseline_label} vs {dst_label}')
    lines.append(f'**评测方法**: ABCD token 概率对比法（与 lm-evaluation-harness 一致）')
    lines.append('')
    lines.append('---')
    lines.append('')

    # 模型信息对比
    lines.append('## 模型信息对比')
    lines.append('')
    lines.append('| 项目 | 基线模型 | DST 模型 |')
    lines.append('|------|---------|---------|')
    lines.append(f'| 标签 | {baseline_label} | {dst_label} |')
    lines.append(f'| 结果文件 | `{os.path.basename(baseline_path)}` | `{os.path.basename(dst_path)}` |')

    # 参数量信息（从结果中推断）
    for dataset_name in ['C-Eval', 'CMMLU']:
        if dataset_name in baseline_results:
            lines.append(f'| {dataset_name} 题目数 | {baseline_results[dataset_name]["total"]} | {dst_results.get(dataset_name, {}).get("total", "-")} |')
            break
    lines.append('')
    lines.append('---')
    lines.append('')

    # 总体能力对比
    lines.append('## 总体能力对比')
    lines.append('')
    lines.append('| 维度 | 基线 | DST | 差异 |')
    lines.append('|------|------|-----|------|')
    for i, dim in enumerate(dimensions):
        b_val = baseline_overall[i]
        d_val = dst_overall[i]
        delta = d_val - b_val
        delta_str = f'{delta:+.2f}pp'
        lines.append(f'| {dim} | {b_val:.2f}% | {d_val:.2f}% | {delta_str} |')
    lines.append('')
    lines.append(f'![总体能力对比](comparison_radar_overall.png)')
    lines.append('')
    lines.append('---')
    lines.append('')

    # C-Eval 子领域对比
    if ceval_sub_labels:
        lines.append('## C-Eval 子领域对比')
        lines.append('')
        lines.append('| 子领域 | 基线 | DST | 差异 |')
        lines.append('|--------|------|-----|------|')
        for i, label in enumerate(ceval_sub_labels):
            delta = d_ceval_vals[i] - b_ceval_vals[i]
            delta_str = f'{delta:+.2f}pp'
            lines.append(f'| {label} | {b_ceval_vals[i]:.2f}% | {d_ceval_vals[i]:.2f}% | {delta_str} |')
        lines.append('')
        lines.append(f'![C-Eval 子领域对比](comparison_radar_ceval.png)')
        lines.append('')
        lines.append('---')
        lines.append('')

    # CMMLU 子领域对比
    if cmmlu_sub_labels:
        lines.append('## CMMLU 子领域对比')
        lines.append('')
        lines.append('| 子领域 | 基线 | DST | 差异 |')
        lines.append('|--------|------|-----|------|')
        for i, label in enumerate(cmmlu_sub_labels):
            delta = d_cmmlu_vals[i] - b_cmmlu_vals[i]
            delta_str = f'{delta:+.2f}pp'
            lines.append(f'| {label} | {b_cmmlu_vals[i]:.2f}% | {d_cmmlu_vals[i]:.2f}% | {delta_str} |')
        lines.append('')
        lines.append(f'![CMMLU 子领域对比](comparison_radar_cmmlu.png)')
        lines.append('')
        lines.append('---')
        lines.append('')

    # 差异分析
    lines.append('## 差异分析')
    lines.append('')
    lines.append('![DST 相对基线差异](comparison_radar_delta.png)')
    lines.append('')

    # 提升最大的科目
    if ceval_deltas:
        lines.append('### C-Eval 差异排名')
        lines.append('')
        lines.append('| 排名 | 科目 | 基线 | DST | 差异 |')
        lines.append('|------|------|------|-----|------|')
        # 提升最大的前 5
        for rank, (task, b_acc, d_acc, delta) in enumerate(ceval_deltas[:5], 1):
            zh = TASK_NAME_ZH.get(task, task)
            lines.append(f'| {rank} | {zh} | {b_acc:.2%} | {d_acc:.2%} | {delta:+.2%} |')
        # 下降最大的前 5
        lines.append('| ... | ... | ... | ... | ... |')
        for rank, (task, b_acc, d_acc, delta) in enumerate(ceval_deltas[-5:], 1):
            zh = TASK_NAME_ZH.get(task, task)
            lines.append(f'| {len(ceval_deltas) - 5 + rank} | {zh} | {b_acc:.2%} | {d_acc:.2%} | {delta:+.2%} |')
        lines.append('')

    if cmmlu_deltas:
        lines.append('### CMMLU 差异排名')
        lines.append('')
        lines.append('| 排名 | 科目 | 基线 | DST | 差异 |')
        lines.append('|------|------|------|-----|------|')
        for rank, (task, b_acc, d_acc, delta) in enumerate(cmmlu_deltas[:5], 1):
            zh = TASK_NAME_ZH.get(task, task)
            lines.append(f'| {rank} | {zh} | {b_acc:.2%} | {d_acc:.2%} | {delta:+.2%} |')
        lines.append('| ... | ... | ... | ... | ... |')
        for rank, (task, b_acc, d_acc, delta) in enumerate(cmmlu_deltas[-5:], 1):
            zh = TASK_NAME_ZH.get(task, task)
            lines.append(f'| {len(cmmlu_deltas) - 5 + rank} | {zh} | {b_acc:.2%} | {d_acc:.2%} | {delta:+.2%} |')
        lines.append('')

    lines.append('---')
    lines.append('')

    # 结论
    lines.append('## 结论')
    lines.append('')
    # 自动生成简要结论
    overall_delta_ceval = dst_results.get('C-Eval', {}).get('accuracy', 0) - baseline_results.get('C-Eval', {}).get('accuracy', 0)
    overall_delta_cmmlu = dst_results.get('CMMLU', {}).get('accuracy', 0) - baseline_results.get('CMMLU', {}).get('accuracy', 0)
    lines.append(f'- C-Eval 总体差异: {overall_delta_ceval:+.2%}（DST {"提升" if overall_delta_ceval > 0 else "下降"}）')
    lines.append(f'- CMMLU 总体差异: {overall_delta_cmmlu:+.2%}（DST {"提升" if overall_delta_cmmlu > 0 else "下降"}）')

    # 统计提升/下降科目数
    if ceval_deltas:
        improved = sum(1 for _, _, _, d in ceval_deltas if d > 0)
        declined = sum(1 for _, _, _, d in ceval_deltas if d < 0)
        lines.append(f'- C-Eval 科目中：{improved} 科提升，{declined} 科下降，{len(ceval_deltas) - improved - declined} 科持平')
    if cmmlu_deltas:
        improved = sum(1 for _, _, _, d in cmmlu_deltas if d > 0)
        declined = sum(1 for _, _, _, d in cmmlu_deltas if d < 0)
        lines.append(f'- CMMLU 科目中：{improved} 科提升，{declined} 科下降，{len(cmmlu_deltas) - improved - declined} 科持平')

    lines.append('')
    lines.append('> 注：四选一随机基线为 25%，两个模型整体表现均接近随机猜测水平。')
    lines.append('> 差异以百分点(pp)表示，正值表示 DST 优于基线，负值表示 DST 弱于基线。')
    lines.append('')

    # 保存 Markdown 报告
    md_path = os.path.join(output_dir, 'comparison_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'对比报告已保存: {md_path}')

    # ---- 7. 保存结构化对比数据 ----
    comparison_data = {
        'baseline_label': baseline_label,
        'dst_label': dst_label,
        'date': date_display,
        'overall': {
            'dimensions': dimensions,
            'baseline': baseline_overall,
            'dst': dst_overall,
            'deltas': [d - b for b, d in zip(baseline_overall, dst_overall)],
        },
    }
    if ceval_sub_labels:
        comparison_data['ceval_subfields'] = {
            'labels': ceval_sub_labels,
            'baseline': b_ceval_vals,
            'dst': d_ceval_vals,
            'deltas': [d - b for b, d in zip(b_ceval_vals, d_ceval_vals)],
        }
    if cmmlu_sub_labels:
        comparison_data['cmmlu_subfields'] = {
            'labels': cmmlu_sub_labels,
            'baseline': b_cmmlu_vals,
            'dst': d_cmmlu_vals,
            'deltas': [d - b for b, d in zip(b_cmmlu_vals, d_cmmlu_vals)],
        }
    if ceval_deltas:
        comparison_data['ceval_task_deltas'] = [
            {'task': t, 'zh': TASK_NAME_ZH.get(t, t), 'baseline': round(b, 4), 'dst': round(d, 4), 'delta': round(delta, 4)}
            for t, b, d, delta in ceval_deltas
        ]
    if cmmlu_deltas:
        comparison_data['cmmlu_task_deltas'] = [
            {'task': t, 'zh': TASK_NAME_ZH.get(t, t), 'baseline': round(b, 4), 'dst': round(d, 4), 'delta': round(delta, 4)}
            for t, b, d, delta in cmmlu_deltas
        ]

    json_path = os.path.join(output_dir, 'comparison_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    print(f'对比数据已保存: {json_path}')

    return md_path, json_path


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description='DST 训练效果对比评测报告生成')
    parser.add_argument('--report_dir', default=None, type=str,
                        help='报告目录（包含基线和 DST 的 JSON 结果文件）')
    parser.add_argument('--baseline', default=None, type=str,
                        help='基线模型结果 JSON 路径（优先于 report_dir + baseline_label 自动拼接）')
    parser.add_argument('--dst', default=None, type=str,
                        help='DST 模型结果 JSON 路径（优先于 report_dir + dst_label 自动拼接）')
    parser.add_argument('--baseline_label', default='baseline', type=str,
                        help='基线模型标签（默认 baseline）')
    parser.add_argument('--dst_label', default='dst', type=str,
                        help='DST 模型标签（默认 dst）')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='输出目录（默认与 report_dir 相同）')
    args = parser.parse_args()

    # 确定结果文件路径
    if args.baseline and args.dst:
        baseline_path = args.baseline
        dst_path = args.dst
        if args.report_dir is None:
            args.report_dir = os.path.dirname(baseline_path)
    elif args.report_dir:
        baseline_path = os.path.join(args.report_dir, f'{args.baseline_label}_benchmark_results.json')
        dst_path = os.path.join(args.report_dir, f'{args.dst_label}_benchmark_results.json')
    else:
        # 默认：查找最新的 reports/ 子目录
        reports_dir = os.path.join('.', 'reports')
        if not os.path.isdir(reports_dir):
            print('错误: 未找到 reports/ 目录，请指定 --report_dir 或 --baseline/--dst')
            return
        subdirs = sorted([d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d))])
        if not subdirs:
            print('错误: reports/ 下无子目录，请指定 --report_dir')
            return
        latest_dir = os.path.join(reports_dir, subdirs[-1])
        args.report_dir = latest_dir
        baseline_path = os.path.join(latest_dir, f'{args.baseline_label}_benchmark_results.json')
        dst_path = os.path.join(latest_dir, f'{args.dst_label}_benchmark_results.json')

    # 检查文件存在性
    if not os.path.exists(baseline_path):
        print(f'错误: 基线结果文件不存在: {baseline_path}')
        return
    if not os.path.exists(dst_path):
        print(f'错误: DST 结果文件不存在: {dst_path}')
        return

    # 输出目录
    if args.output_dir is None:
        args.output_dir = args.report_dir

    print('=' * 60)
    print('DST 训练效果对比评测报告生成')
    print('=' * 60)
    print(f'基线结果: {baseline_path}')
    print(f'DST 结果: {dst_path}')
    print(f'输出目录: {args.output_dir}')

    generate_comparison_report(baseline_path, dst_path, args.baseline_label, args.dst_label, args.output_dir)

    print('\n对比报告生成完成!')


if __name__ == '__main__':
    main()
