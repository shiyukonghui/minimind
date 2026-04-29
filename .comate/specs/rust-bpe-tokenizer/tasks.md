# Rust + wordchipper 复刻 BPE Tokenizer 训练功能

- [x] Task 1: 更新 Cargo.toml 依赖配置
    - 1.1: 确认现有依赖（wordchipper, wordchipper-training, serde, serde_json, anyhow, log, env_logger）是否充足
    - 1.2: 如需补充依赖（如内存优化的 JSONL 读取），添加到 Cargo.toml

- [x] Task 2: 实现 JSONL 数据读取模块
    - 2.1: 实现逐行读取 JSONL 文件的函数
    - 2.2: 解析每行 JSON 提取 `text` 字段
    - 2.3: 处理文件不存在和行解析失败的异常情况

- [x] Task 3: 实现 BPE 训练核心逻辑
    - 3.1: 配置特殊 token（`<pad>`, `<|im_start|>`, `<|im_end|>`）及其 ID
    - 3.2: 使用 `OA_GPT2_PATTERN_SLOW` 初始化 BPETrainer（vocab_size=6400）
    - 3.3: 将 JSONL 文本数据分批喂入 trainer
    - 3.4: 调用 `trainer.train()` 获取 UnifiedTokenVocab
    - 3.5: 断言特殊 token ID 正确（0, 1, 2）

- [x] Task 4: 实现 ByteLevel 字节编码转换
    - 4.1: 实现 `byte_to_bytelevel_char()` 函数（可打印 ASCII 直接映射，其他字节映射到 U+0100 + byte_value）
    - 4.2: 实现 `bytes_to_bytelevel_string()` 函数，将 `Vec<u8>` 转换为 HuggingFace ByteLevel 字符串表示

- [x] Task 5: 实现 HuggingFace tokenizer.json 导出
    - 5.1: 从 UnifiedTokenVocab 提取 unified_dictionary 构建词汇表映射
    - 5.2: 从 pair_vocab 提取 merges 列表，按 merged_id 排序还原训练顺序
    - 5.3: 构建 added_tokens 数组（3 个特殊 token）
    - 5.4: 组装完整 JSON 结构（version, truncation, padding, pre_tokenizer, decoder, model 等）
    - 5.5: 序列化并写入 `model/tokenizer.json`

- [x] Task 6: 实现 tokenizer_config.json 导出
    - 6.1: 使用 serde_json 手动构建与 Python 脚本一致的配置对象
    - 6.2: 包含 added_tokens_decoder、chat_template 等完整字段
    - 6.3: 序列化并写入 `model/tokenizer_config.json`

- [x] Task 7: 组装 main 函数并验证编译
    - 7.1: 在 main 函数中串联完整流程（读取 → 训练 → 导出）
    - 7.2: 添加日志输出（训练进度、文件保存路径等）
    - 7.3: 运行 `cargo build` 确认编译通过
