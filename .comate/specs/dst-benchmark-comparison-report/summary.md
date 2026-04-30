# DST 基线对比评测报告 — 实施总结

## 完成内容

### 1. 修改 eval_benchmark.py 报告文件名逻辑

- 新增 `infer_model_label()` 函数，从模型路径自动推断标签（dst/pretrain/默认取路径名）
- 修改 `generate_report()` 函数，新增 `model_label` 参数，输出文件名改为 `{model_label}_benchmark_results.json` 和 `{model_label}_eval_report.md`
- 新增 `--model_label` CLI 参数，允许手动指定标签
- 修改 `main()` 函数，传递 `model_label` 给 `generate_report()`

### 2. 创建 eval_compare.py 对比报告脚本

完整实现了以下功能：

- **数据加载**：`load_results()` 从 JSON 文件加载评测结果
- **数据聚合**：`compute_category_accuracy()`、`compute_subfield_accuracies()`、`compute_overall_dimensions()`、`compute_task_deltas()`
- **子领域映射**：定义了 `CEVAL_SUBFIELDS`（6 子领域）和 `CMMLU_SUBFIELDS`（6 子领域）
- **雷达图绘制**：`draw_radar_chart()` 双模型对比雷达图 + `draw_delta_radar()` 差异雷达图
- **对比报告生成**：`generate_comparison_report()` 生成 Markdown 报告 + 结构化 JSON 数据
- **中文字体兼容**：`setup_chinese_font()` 自动尝试 SimHei 等中文字体

### 3. 验证结果

使用 `reports/20260429/`（基线）和 `reports/20260430/`（DST）的数据成功运行，生成：

| 文件 | 说明 |
|------|------|
| `comparison_radar_overall.png` | 总体能力对比雷达图（6 维） |
| `comparison_radar_ceval.png` | C-Eval 子领域雷达图（6 维） |
| `comparison_radar_cmmlu.png` | CMMLU 子领域雷达图（6 维） |
| `comparison_radar_delta.png` | DST 相对基线差异雷达图 |
| `comparison_report.md` | 完整对比报告 |
| `comparison_data.json` | 结构化对比数据（944 行） |

### 关键发现（来自对比报告）

- C-Eval 总体：DST 下降 0.89pp（基线 23.33% → DST 22.44%）
- CMMLU 总体：DST 提升 0.25pp（基线 25.09% → DST 25.34%）
- C-Eval 科目中：17 科提升，22 科下降，13 科持平
- CMMLU 科目中：35 科提升，26 科下降，6 科持平
- DST 在 STEM 方面略有提升（CMMLU +0.23pp），人文社科表现波动较大

## 使用方式

```bash
# 1. 运行基线评测（输出到同目录，文件名加前缀）
python eval_benchmark.py --model_path DST-train/model/baseline_hf --model_label baseline

# 2. 运行 DST 评测（同目录不同文件名）
python eval_benchmark.py --model_path DST-train/model/dst_hf --model_label dst

# 3. 生成对比报告
python eval_compare.py --report_dir reports/20260430 --baseline_label baseline --dst_label dst
```
