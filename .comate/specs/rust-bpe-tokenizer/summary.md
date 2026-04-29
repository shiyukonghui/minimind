# Rust + wordchipper 复刻 BPE Tokenizer 训练功能 - 完成总结

## 任务概述

使用 Rust + wordchipper/wordchipper-training 库，在 `rust-tokenizer/` 目录下复刻 `scripts/train_tokenizer.py` 的功能，生成与 HuggingFace tokenizers 格式兼容的 `tokenizer.json` 和 `tokenizer_config.json`。

## 完成情况

所有 7 个任务均已完成。

## 关键实现细节

### 1. 依赖配置 (Cargo.toml)
- 移除了 `wordchipper` 的 `client` feature（不需要下载预训练模型）
- 添加了 `critical-section = { version = "1", features = ["std"] }` 解决 Windows 链接错误
- 现有依赖 `serde`, `serde_json`, `anyhow`, `log`, `env_logger` 均已使用

### 2. JSONL 数据读取
- 使用 `BufReader` 逐行读取 `dataset/pretrain_t2t_mini.jsonl`
- 解析每行 JSON 提取 `text` 字段，跳过空行和解析失败的行
- 使用 `CARGO_MANIFEST_DIR` 环境变量定位项目根目录，避免运行时路径问题

### 3. BPE 训练
- 使用 `OA_GPT2_PATTERN_SLOW` 预分词器（与 HuggingFace ByteLevel 正则完全等价）
- `vocab_size` 设为 6397（6400 - 3 特殊 token），因为 wordchipper 的 byte vocab 占用 ID 0-255
- 训练后通过 ID 偏移（+3）将 wordchipper ID 映射到 HuggingFace ID（特殊 token 占 0-2，byte vocab 占 3-258，BPE 合并从 259 开始）

### 4. ByteLevel 编码转换
- 可打印 ASCII (0x21-0x7E) 直接映射为自身
- 其他字节映射到 Unicode 码点 U+0100 + byte_value（与 GPT-2 ByteLevel 编码一致）

### 5. HuggingFace tokenizer.json 导出
- 从 `UnifiedTokenVocab` 的 `unified_dictionary()` 提取 token -> bytes 映射
- **关键问题及解决方案**：`pair_vocab()` 通过 `SpanMapVocab::to_pair_vocab()` 重新计算，包含所有可能的二分分解（而非仅 BPE 规范合并），导致 merges 数量从 6141 膨胀到 7471
- **解决方案**：实现 `extract_canonical_merges()` 函数，从 `span_vocab` 直接推导 BPE 规范合并：
  - 对每个非字节 token，遍历其字节序列的所有二分位置
  - 筛选两个子 token ID 都小于当前 token ID 的分解（BPE 性质）
  - 如有多个有效分解，选择子 token ID 之和最大的（对应最后一步合并）
  - 按合并后的 token ID 排序还原训练顺序
- 使用 `serde_json::Map` 手动控制 JSON 字段顺序

### 6. tokenizer_config.json 导出
- 从已有的 `tokenizer_config.json` 读取 `chat_template`（避免在 Rust 代码中硬编码含 XML 标签的 Jinja2 模板）
- 使用 `serde_json::Map` 逐步构建配置对象

### 7. 验证结果
- 编译通过（cargo build）
- 训练成功完成（6141 merges, 6400 vocab）
- 特殊 token ID 验证通过（`<pad>=0`, `<|im_start|>=1`, `<|im_end|>=2`）
- 词汇表 ID 从 0 到 6399 连续
- 使用 `tokenizers.Tokenizer.from_file()` 加载成功
- 编码 "hello world" -> `[380, 860, 114, 632, 500, 1729]`，解码后与原文一致

## 修改文件清单

| 文件 | 修改类型 |
|------|---------|
| `rust-tokenizer/src/main.rs` | 重写：完整训练+导出逻辑 |
| `rust-tokenizer/Cargo.toml` | 修改：移除 client feature，添加 critical-section |

## 已知限制

1. **训练时间较长**：release 模式下约 13 分钟（Python 版本约 2 分钟），主要瓶颈在 wordchipper-training 的 BPE 合并循环
2. **BPE 合并顺序近似**：`extract_canonical_merges()` 使用"子 token ID 之和最大"启发式规则，在极少数情况下可能与原始训练顺序不同
3. **chat_template 依赖外部文件**：首次运行前需确保 `model/tokenizer_config.json` 包含正确的 chat_template
