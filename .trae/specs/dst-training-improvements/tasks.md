# Tasks

- [x] Task 1: 新增 `GNDMonitor` 类替代MBE作为饱和检测器
  - [x] 1.1 在 `dst_hooks.py` 中新增 `GNDMonitor` 类，包含以下方法：
    - `__init__(self, model, threshold=0.1, patience=5)` — 初始化，记录peak为None，low_count为0
    - `compute_grad_norm(self)` — 遍历所有 `param.grad` 非None的2D权重，计算全局梯度L2范数 `sqrt(sum(||grad||_2^2))`
    - `check_saturation(self)` — 调用 `compute_grad_norm`，首次调用时记录 `peak`，之后计算 `ratio = cur / peak`，若 `ratio < threshold` 则 `low_count += 1` 否则 `low_count = 0`，返回 `(is_saturated, ratio, grad_norm)`
    - `report(self)` — 返回当前GND状态的字符串摘要
  - [x] 1.2 确认 `GNDMonitor` 在梯度累积后（`optimizer.step()` 之后、`zero_grad()` 之前）被调用，此时梯度已累积完成

- [x] Task 2: 在 `train_dst.py` 中集成GND监控并实现阶段一提前终止
  - [x] 2.1 新增 `--phase1_max_epochs` 参数，默认值为 `args.phase1_epochs`
  - [x] 2.2 新增 `--gnd_threshold` 参数，默认值为 0.1
  - [x] 2.3 新增 `--gnd_patience` 参数，默认值为 3
  - [x] 2.4 在 `train_dst.py` 中从 `dst_hooks` 导入 `GNDMonitor`
  - [x] 2.5 在阶段一初始化处创建 `GNDMonitor` 实例
  - [x] 2.6 在阶段一训练循环中，梯度累积完成后（`optimizer.step()` 之后、`zero_grad()` 之前）调用 `gnd_monitor.check_saturation()`
  - [x] 2.7 若 `is_saturated=True`，打印日志并 `break` 跳出循环
  - [x] 2.8 阶段一循环结束后，根据退出原因打印日志（饱和退出 vs 达到最大轮数）
  - [x] 2.9 在阶段一训练结束后打印GND诊断报告
  - [x] 2.10 将MBE监控从饱和判定降级为纯诊断日志（保留 `check_model_mbe()` 调用但仅打印 `[DST-MBE-Diag]` 前缀日志，不使用其 `is_saturated` 结果）

- [x] Task 3: 公平训练轮次 — 阶段三默认跳过
  - [x] 3.1 在 `train_dst.py` 中将 `--phase3_epochs` 默认值从 3 → 0
  - [x] 3.2 在阶段三入口处增加条件判断：若 `phase3_epochs <= 0` 则跳过整个阶段三
  - [x] 3.3 跳过阶段三时打印日志："[DST] phase3_epochs=0，跳过恢复训练阶段"

- [x] Task 4: 增大batch size加快训练
  - [x] 4.1 在 `train_dst.py` 中将 `--phase1_batch_size` 默认值从 32 → 300
  - [x] 4.2 在 `train_dst.py` 中将 `--phase1_accumulation_steps` 默认值从 8 → 1
  - [x] 4.3 在 `train_baseline.py` 中将 `--batch_size` 默认值从 32 → 300
  - [x] 4.4 在 `train_baseline.py` 中将 `--accumulation_steps` 默认值从 8 → 1

# Task Dependencies
- [Task 2] 依赖于 [Task 1]（需先完成GNDMonitor类才能集成）
- [Task 3] 和 [Task 4] 互不依赖，可并行执行，且均独立于 [Task 1] [Task 2]
