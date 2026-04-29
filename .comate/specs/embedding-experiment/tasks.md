# 多种向量化表示方法实验 - 任务计划

- [x] Task 1: 创建 model_embedding.py（多嵌入方案模型定义）
    - 1.1: 定义 MiniMindEmbeddingConfig，继承 MiniMindConfig，新增 embed_type 和 coord_dim 参数
    - 1.2: 实现 generate_fixed_2d_coords 函数（方案B二维固定坐标生成）
    - 1.3: 实现 generate_fixed_3d_sinusoid 函数（方案C-1三维正弦编码生成）
    - 1.4: 实现 MiniMindEmbeddingModel，继承 MiniMindModel，重写嵌入层（A/B/C1/C2 四分支）
    - 1.5: 实现 MiniMindEmbeddingForCausalLM，继承 MiniMindForCausalLM，处理权重绑定逻辑（仅A绑定，B/C1/C2断开）
    - 1.6: 验证模型能正常前向传播，各方案参数量统计正确

- [x] Task 2: 创建 train_embedding_exp.py（统一训练脚本）
    - 2.1: 基于 train_baseline.py 搭建训练框架，新增 --embed_type 参数（B/C1/C2）
    - 2.2: 实现 init_embedding_model 函数，根据 embed_type 创建 MiniMindEmbeddingForCausalLM 实例
    - 2.3: 添加嵌入参数量统计与日志输出
    - 2.4: 模型保存路径包含方案标识：{save_dir}/{embed_type}_512.pth
    - 2.5: 训练日志标注方案名称，便于区分

- [x] Task 3: 创建 eval_embedding_exp.py（评测+语义分析脚本）
    - 3.1: 实现基准评测功能（C-Eval + CMMLU），复用 eval_benchmark.py 的 ABCD token 概率对比法
    - 3.2: 实现嵌入权重提取函数，根据 embed_type 从不同层获取嵌入表示
    - 3.3: 实现高频字最邻近分析（余弦相似度 top-5 邻居）
    - 3.4: 实现 t-SNE 语义拓扑可视化（降维+散点图+汉字标注）
    - 3.5: 实现参数效率比计算（语义耦合度 = PPL / 嵌入参数量）
    - 3.6: 实现训练 loss 曲线对比图绘制
    - 3.7: 实现四组方案汇总对比报告生成

- [x] Task 4: 创建 convert_for_eval.py（格式转换脚本）
    - 4.1: 基于 DST-train/convert_for_eval.py 搭建转换框架
    - 4.2: 支持 --embed_type 参数，使用 MiniMindEmbeddingConfig 和 MiniMindEmbeddingForCausalLM
    - 4.3: 输出目录命名：{model_dir}/{embed_type}_hf
    - 4.4: 自动复制分词器到输出目录

- [x] Task 5: 创建 run_all.bat 和 run_all.sh（全流程脚本）
    - 5.1: Step1 - 评测基线方案A（直接使用 MiniMind2-Pretrain-512，无需训练）
    - 5.2: Step2 - 训练方案B（embed_type=B）
    - 5.3: Step3 - 训练方案C1（embed_type=C1）
    - 5.4: Step4 - 训练方案C2（embed_type=C2）
    - 5.5: Step5 - 转换B/C1/C2模型为HuggingFace格式
    - 5.6: Step6 - 四组方案基准评测（C-Eval + CMMLU）
    - 5.7: Step7 - 四组方案语义分析对比
    - 5.8: 支持跳过已完成的步骤（检测模型文件是否存在）
