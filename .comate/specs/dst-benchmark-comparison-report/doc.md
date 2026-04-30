# DST 基线对比评测报告 — 需求文档

## 需求场景

DST（动态稀疏训练）实验已完成基线模型（pretrain_512）和 DST 恢复模型（dst_hf）的独立评测。目前存在两个问题：

1. **报告路径冲突**：`eval_benchmark.py` 按 `reports/YYYYMMDD/` 组织报告，同日评测不同模型会产生同名文件互相覆盖（如 `benchmark_results.json` 和 `eval_report.md` 在同日期目录下会冲突）
2. **缺少对比报告**：需要一份将基线和 DST 汇总对比的报告，以**雷达图**形式直观呈现差异

## 解决方案

### 问题1：修改 eval_benchmark.py 的报告命名

修改 `generate_report()` 函数，在文件名中加入模型标签以避免覆盖：

- 原命名：`benchmark_results.json` / `eval_report.md`
- 新命名：`{model_label}_benchmark_results.json` / `{model_label}_eval_report.md`

`model_label` 由 `--model_label` CLI 参数指定，默认从模型路径自动推断（如 `baseline`、`dst`）。

目录结构示例：
```
reports/20260430/
├── baseline_benchmark_results.json
├── baseline_eval_report.md
├── dst_benchmark_results.json
├── dst_eval_report.md
├── comparison_radar_overall.png
├── comparison_radar_ceval.png
├── comparison_radar_cmmlu.png
├── comparison_radar_delta.png
├── comparison_report.md
└── comparison_data.json
```

### 问题2：新增 eval_compare.py 对比报告脚本

读取同目录下的基线和 DST 结果文件，生成对比报告和雷达图。

## 架构与技术方案

### 修改文件

- `f:\MachineLearn\minimind\eval_benchmark.py` — 修改报告文件名逻辑，加入 model_label

### 新增文件

- `f:\MachineLearn\minimind\eval_compare.py` — 对比评测报告生成脚本

### 依赖

- `matplotlib` — 绘制雷达图
- 现有 `eval_benchmark.py` 中的 `STEM_TASKS`、`HUMANITIES_TASKS`、`TASK_NAME_ZH` 常量（从 eval_compare.py 中导入或复制）

### 数据来源

两个 JSON 文件位于同一报告目录下，通过文件名区分：
- `{baseline_label}_benchmark_results.json`
- `{dst_label}_benchmark_results.json`

## eval_benchmark.py 修改细节

### 修改 `generate_report()` 函数

```python
def generate_report(results, model_path, model_params_m=None, model_label=None, output_dir=None):
    # model_label: 模型标签，用于文件名前缀
    # 默认从模型路径推断：dst_hf -> dst, pretrain_512 -> baseline, MiniMind2 -> minimind2
    if model_label is None:
        model_name = os.path.basename(model_path.rstrip('/\\'))
        # 推断标签
        if 'dst' in model_name.lower():
            model_label = 'dst'
        elif 'pretrain' in model_name.lower():
            model_label = 'baseline'
        else:
            model_label = model_name

    # 文件名加前缀
    json_path = os.path.join(output_dir, f'{model_label}_benchmark_results.json')
    md_path = os.path.join(output_dir, f'{model_label}_eval_report.md')
```

### 修改 CLI 参数

新增 `--model_label` 参数：
```python
parser.add_argument('--model_label', default=None, type=str, help='模型标签（用于报告文件名前缀，默认从模型路径推断）')
```

### 修改 `main()` 函数

将 `model_label` 传递给 `generate_report()`。

## eval_compare.py 实现细节

### CLI 接口

```bash
python eval_compare.py \
    --report_dir reports/20260430 \
    --baseline_label baseline \
    --dst_label dst \
    --output_dir reports/20260430
```

默认行为：`--report_dir` 与 `--output_dir` 相同，对比报告和雷达图与原始评测结果放在一起。

### 核心函数

```python
def load_results(json_path: str) -> dict:
    """加载 benchmark_results.json"""

def compute_category_accuracy(result: dict, tasks: list[str]) -> float:
    """计算指定科目列表的聚合准确率"""

def compute_subfield_accuracy(result: dict, subfield_tasks: dict[str, list[str]]) -> dict[str, float]:
    """计算子领域准确率映射"""

def draw_radar_chart(labels, values_dict, title, save_path):
    """绘制雷达图"""

def generate_comparison_report(report_dir, baseline_label, dst_label, output_dir):
    """主函数：加载数据、计算、绘图、生成 Markdown 报告"""
```

### 雷达图设计

**图1：总体能力雷达图（6 维）**
- 维度：C-Eval 总准确率、C-Eval STEM、C-Eval 人文社科、CMMLU 总准确率、CMMLU STEM、CMMLU 人文社科

**图2：C-Eval 子领域雷达图（6 维）**
- 数学、物理、化学/生物/医学、计算机、工程/应用、人文社科

**图3：CMMLU 子领域雷达图（6 维）**
- 数学/逻辑、物理/工程、化学/生物/医学、计算机、人文/社科/法律、经济/管理/教育

**图4：差异雷达图（Delta Radar，6 维）**
- 与图1相同维度，展示 DST 相对基线的百分比差异

### 雷达图绘制要点

- 使用 `matplotlib.pyplot` 极坐标子图 (`subplot(polar=True)`)
- 角度均分，闭合多边形
- 基线模型用蓝色，DST 模型用红色
- 半透明填充（alpha=0.25）
- 百分比刻度格式
- DPI=150，支持中文标题（尝试 SimHei 字体，回退到默认）

### 子领域科目映射

**C-Eval 子领域：**
```python
CEVAL_SUBFIELDS = {
    '数学': ['advanced_mathematics', 'high_school_mathematics', 'middle_school_mathematics',
             'discrete_mathematics', 'probability_and_statistics'],
    '物理': ['college_physics', 'high_school_physics', 'middle_school_physics'],
    '化学/生物/医学': ['college_chemistry', 'high_school_chemistry', 'middle_school_chemistry',
                     'high_school_biology', 'middle_school_biology', 'basic_medicine',
                     'clinical_medicine', 'veterinary_medicine', 'plant_protection'],
    '计算机': ['college_programming', 'computer_architecture', 'computer_network', 'operating_system'],
    '工程/应用': ['electrical_engineer', 'metrology_engineer', 'fire_engineer',
               'environmental_impact_assessment_engineer'],
    '人文社科': ['law', 'legal_professional', 'marxism', 'mao_zedong_thought',
              'ideological_and_moral_cultivation', 'high_school_politics', 'middle_school_politics',
              'high_school_history', 'middle_school_history', 'modern_chinese_history',
              'education_science', 'teacher_qualification', 'civil_servant', 'accountant',
              'tax_accountant', 'business_administration', 'urban_and_rural_planner',
              'physician', 'professional_tour_guide', 'sports_science', 'art_studies',
              'chinese_language_and_literature', 'high_school_chinese', 'high_school_geography',
              'middle_school_geography', 'middle_school_chemistry', 'logic',
              'college_economics'],
}
```

**CMMLU 子领域：**
```python
CMMLU_SUBFIELDS = {
    '数学/逻辑': ['college_mathematics', 'elementary_mathematics', 'college_actuarial_science',
                'logical', 'machine_learning'],
    '物理/工程': ['conceptual_physics', 'high_school_physics', 'electrical_engineering',
               'college_engineering_hydrology', 'construction_project_management'],
    '化学/生物/医学': ['anatomy', 'genetics', 'virology', 'nutrition', 'college_medicine',
                     'professional_medicine', 'clinical_knowledge', 'traditional_chinese_medicine',
                     'food_science'],
    '计算机': ['computer_science', 'computer_security'],
    '人文/社科/法律': ['chinese_history', 'world_history', 'ancient_chinese', 'modern_chinese',
                   'chinese_literature', 'philosophy', 'sociology', 'journalism',
                   'public_relations', 'international_law', 'college_law', 'professional_law',
                   'jurisprudence', 'legal_and_moral_basis'],
    '经济/管理/教育': ['economics', 'management', 'marketing', 'business_ethics',
                   'education', 'college_education', 'chinese_teacher_qualification',
                   'professional_psychology', 'professional_accounting'],
}
```

### 对比报告结构

```markdown
# DST 训练效果对比评测报告

## 模型信息对比
| 项目 | 基线模型 | DST 模型 |
|------|---------|---------|
| 模型路径 | ... | ... |
| 参数量 | ... | ... |

## 总体能力对比
（表格 + 图1 雷达图）

## C-Eval 子领域对比
（表格 + 图2 雷达图）

## CMMLU 子领域对比
（表格 + 图3 雷达图）

## 差异分析
（提升/下降最大的科目 + 图4 Delta 雷达图）

## 结论
```

## 边界条件与异常处理

1. **matplotlib 未安装**：捕获 ImportError，提示 `pip install matplotlib`
2. **结果文件不存在**：明确报错并退出
3. **科目列表不匹配**：取两份结果中共同存在的科目进行对比
4. **子领域无数据**：跳过或标记为 N/A
5. **中文字体缺失**：尝试 SimHei，回退到默认字体，不因字体问题中断

## 数据流路径

```
eval_benchmark.py --model_label baseline → reports/YYYYMMDD/baseline_benchmark_results.json
eval_benchmark.py --model_label dst     → reports/YYYYMMDD/dst_benchmark_results.json
                                                    │
eval_compare.py --report_dir reports/YYYYMMDD ──────┘
    ├── 读取 baseline_benchmark_results.json
    ├── 读取 dst_benchmark_results.json
    ├── 计算聚合维度准确率
    ├── 绘制雷达图 PNG
    └── 生成 comparison_report.md + comparison_data.json
```

## 预期产出

所有产出位于同一 `reports/YYYYMMDD/` 目录下：

1. `baseline_benchmark_results.json` / `baseline_eval_report.md` — 基线评测
2. `dst_benchmark_results.json` / `dst_eval_report.md` — DST 评测
3. `comparison_radar_overall.png` — 总体能力雷达图
4. `comparison_radar_ceval.png` — C-Eval 子领域雷达图
5. `comparison_radar_cmmlu.png` — CMMLU 子领域雷达图
6. `comparison_radar_delta.png` — 差异雷达图
7. `comparison_report.md` — 完整对比报告
8. `comparison_data.json` — 结构化对比数据
