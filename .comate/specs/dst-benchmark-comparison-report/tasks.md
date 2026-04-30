# DST 基线对比评测报告 — 任务计划

- [x] Task 1: 修改 eval_benchmark.py 报告文件名逻辑
    - 1.1: 修改 `generate_report()` 函数，新增 `model_label` 参数，用于文件名前缀
    - 1.2: 在 `generate_report()` 中实现 model_label 自动推断逻辑（dst/pretrain/默认取路径名）
    - 1.3: 修改 JSON 和 Markdown 输出文件名为 `{model_label}_benchmark_results.json` / `{model_label}_eval_report.md`
    - 1.4: 新增 CLI 参数 `--model_label`
    - 1.5: 修改 `main()` 函数，将 model_label 传递给 `generate_report()`

- [x] Task 2: 创建 eval_compare.py 对比脚本骨架
    - 2.1: 定义 CLI 参数（report_dir, baseline_label, dst_label, output_dir）
    - 2.2: 实现 `load_results()` 函数加载 JSON 结果
    - 2.3: 复用 eval_benchmark.py 中的 STEM_TASKS、HUMANITIES_TASKS、TASK_NAME_ZH 常量

- [x] Task 3: 实现数据聚合与对比计算
    - 3.1: 实现 `compute_category_accuracy()` 函数（按学科大类聚合）
    - 3.2: 定义 CEVAL_SUBFIELDS 和 CMMLU_SUBFIELDS 子领域映射
    - 3.3: 实现 `compute_subfield_accuracy()` 函数（按子领域聚合）
    - 3.4: 实现对比数据计算逻辑（差异百分比、提升/下降科目排名）

- [x] Task 4: 实现雷达图绘制功能
    - 4.1: 实现 `draw_radar_chart()` 通用雷达图绘制函数
    - 4.2: 处理中文字体兼容性（SimHei 尝试 + 回退）
    - 4.3: 绘制总体能力雷达图（6 维）
    - 4.4: 绘制 C-Eval 子领域雷达图（6 维）
    - 4.5: 绘制 CMMLU 子领域雷达图（6 维）
    - 4.6: 绘制差异雷达图（Delta Radar）

- [x] Task 5: 实现对比报告生成
    - 5.1: 实现模型信息对比表格
    - 5.2: 实现总体能力对比表格 + 雷达图嵌入
    - 5.3: 实现 C-Eval / CMMLU 子领域对比表格 + 雷达图嵌入
    - 5.4: 实现差异分析（提升/下降最大科目排名）+ Delta 雷达图嵌入
    - 5.5: 生成 comparison_data.json 结构化对比数据
    - 5.6: 实现 `main()` 函数串联完整流程

- [x] Task 6: 验证与测试
    - 6.1: 使用现有 reports/20260429 和 reports/20260430 的数据重命名后运行 eval_compare.py
    - 6.2: 验证雷达图 PNG 生成正确
    - 6.23: 验证对比报告 Markdown 内容完整
