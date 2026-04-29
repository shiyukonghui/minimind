# MiniMind 模型性能评分测试任务计划

- [x] Task 1: 安装 lm-evaluation-harness 依赖
    - 1.1: 使用 uv 在 .venv 环境中安装 lm-evaluation-harness 及其依赖
    - 1.2: 验证 lm_eval 命令行工具可用

- [x] Task 2: 将 PyTorch 模型转换为 HuggingFace 格式
    - 2.1: 修改 scripts/convert_model.py 中的路径参数，指向 MiniMind2-PyTorch/full_sft_768.pth
    - 2.2: 修正 convert_model.py 中 MiniMindConfig 的参数（移除不存在的 max_seq_len）
    - 2.3: 执行转换脚本，输出 HuggingFace Llama 格式模型到 MiniMind2 目录
    - 2.4: 验证转换后的模型文件完整性（config.json、pytorch_model.bin、tokenizer 文件）

- [x] Task 3: 使用 lm_eval 运行 C-Eval 评测
    - 3.1: 执行 lm_eval 命令对 MiniMind2 模型进行 ceval_valid 评测
    - 3.2: 记录各学科类别的准确率结果

- [x] Task 4: 使用 lm_eval 运行 CMMLU 评测
    - 4.1: 执行 lm_eval 命令对 MiniMind2 模型进行 cmmlu 评测
    - 4.2: 记录各学科类别的准确率结果

- [x] Task 5: 汇总评测结果
    - 5.1: 整理 C-Eval 和 CMMLU 的评测数据
    - 5.2: 输出格式化的评分报告
