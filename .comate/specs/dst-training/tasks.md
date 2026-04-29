# DST 动态稀疏训练 - 任务计划

- [x] Task 1: 创建 DST-train 目录结构及模型输出目录
    - 1.1: 创建 `DST-train/` 顶级目录
    - 1.2: 创建 `DST-train/model/` 子目录用于存放训练产出

- [x] Task 2: 编写 dst_hooks.py 诊断Hook模块
    - 2.1: 实现 `MBEMonitor` 类 — 矩阵基熵(MBE)计算、饱和度判断、Hook注册与移除、稀疏度统计
    - 2.2: 实现 `DeadNeuronDetector` 类 — 假死神经元检测、唤醒策略(wake_dead_neurons)

- [x] Task 3: 编写 dst_pruning.py 剪枝器模块
    - 3.1: 实现 `MagnitudePruner` 类 — compute_mask计算剪枝掩码、apply_mask应用掩码、稀疏度统计报告
    - 3.2: 实现 `RigLScheduler` 类 — 初始稀疏掩码创建、基于梯度的剪枝-生长更新(update方法)、T_end后固定掩码

- [x] Task 4: 编写 train_baseline.py 基线模型训练脚本
    - 4.1: 基于现有 train_pretrain.py 逻辑，改造为从头训练(from_weight='none')的独立脚本
    - 4.2: 配置命令行参数，保存权重到 `DST-train/model/baseline_512.pth`
    - 4.3: 训练循环：余弦退火学习率、混合精度、梯度累积、日志与保存

- [x] Task 5: 编写 train_dst.py DST训练主脚本
    - 5.1: 阶段一实现 — 从头预训练 + RigL动态稀疏训练，每个训练步应用稀疏掩码
    - 5.2: 阶段二实现 — 调用MagnitudePruner计算并应用剪枝掩码，保存剪枝前模型作为教师
    - 5.3: 阶段三实现 — SFT微调恢复 + 可选知识蒸馏 + 假死神经元唤醒
    - 5.4: 命令行参数定义（通用、阶段一/二/三、DST专用参数）

- [x] Task 6: 编写 convert_for_eval.py 模型格式转换脚本
    - 6.1: 实现 PyTorch 权重 → HuggingFace 格式转换，支持 baseline/dst_phase1/dst_recovered 三种模型类型
    - 6.2: 自动复制分词器文件到输出目录

- [x] Task 7: 编写 run_eval.sh 一键测评脚本
    - 7.1: 依次评测基线模型和DST模型，输出对比结果
