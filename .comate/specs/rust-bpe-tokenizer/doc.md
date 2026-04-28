# 使用 Rust + wordchipper 复刻 BPE Tokenizer 训练功能

## 需求场景

原项目使用 Python (`scripts/train_tokenizer.py`) 基于 HuggingFace `tokenizers` 库训练 BPE 分词器。现需在 `rust-tokenizer/` 目录下使用 Rust + `wordchipper` / `wordchipper-training` 库复刻同等功能，产出与原 Python 脚本**格式兼容**的 `tokenizer.json` 和 `tokenizer_config.json`。

## 原始 Python 逻辑分析

`train_tokenizer.py` 的核心流程：

1. **读取训练数据**：从 `dataset/pretrain_t2t_mini.jsonl` 逐行读取 JSON，提取 `text` 字段
2. **初始化 BPE 分词器**：使用 `Tokenizer(models.BPE())` + `ByteLevel(add_prefix_space=False)` 预分词器
3. **配置训练器**：`BpeTrainer(vocab_size=6400, special_tokens=["", "<|im_start|>", "<|im_end|>"], initial_alphabet=ByteLevel.alphabet())`
4. **训练**：`tokenizer.train_from_iterator(texts, trainer=trainer)`
5. **设置解码器**：`tokenizer.decoder = decoders.ByteLevel()`
6. **断言特殊 token ID**：验证 `<pad>=0`, `<|im_start|>=1`, `<|im_end|>=2`
7. **保存 `tokenizer.json`**：HuggingFace tokenizers 标准格式
8. **保存 `tokenizer_config.json`**：手动构建的 HuggingFace transformers 配置

## 架构与技术方案

### 技术选型

- **BPE 训练**：`wordchipper-training` 的 `BPETrainer`
- **预分词器**：`wordchipper::pretrained::openai::OA_GPT2_PATTERN_SLOW`（与 HuggingFace ByteLevel 正则完全等价）
- **特殊 token**：通过 `TextSpanningConfig::with_special_words()` 设置
- **JSON 输出**：`wordchipper` 无内置 HuggingFace 格式导出，需手动构建 JSON 结构

### 数据流

```
JSONL 文件 → 逐行解析提取 text → BPETrainer.update_from_samples()
    → trainer.train(byte_vocab) → UnifiedTokenVocab
    → 从 UnifiedTokenVocab 提取 vocab / merges / specials
    → 手动构建 HuggingFace tokenizer.json 格式
    → 手动构建 tokenizer_config.json
    → 写入 model/ 目录
```

## 受影响文件

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `rust-tokenizer/src/main.rs` | 重写 | 主逻辑：读取数据、训练、导出 |
| `rust-tokenizer/Cargo.toml` | 修改 | 可能需要添加依赖（如 `regex` 等） |

## 实现细节

### 1. 特殊 Token 设置

```rust
type T = u32;

let special_tokens: Vec<(&str, T)> = vec![
    ("", 0),
    ("<|im_start|>", 1),
    ("<|im_end|>", 2),
];
```

### 2. 训练流程

```rust
use wordchipper::pretrained::openai::OA_GPT2_PATTERN_SLOW;
use wordchipper::spanners::TextSpanningConfig;
use wordchipper::vocab::ByteMapVocab;
use wordchipper_training::{BPETRainerOptions, BPETrainer};

let spanning_config = TextSpanningConfig::<T>::from_pattern(OA_GPT2_PATTERN_SLOW)
    .with_special_words(special_tokens.clone());

let mut trainer = BPETRainerOptions::new(OA_GPT2_PATTERN_SLOW, 6400).init();

// 读取 JSONL 数据，批量喂入
for line in jsonl_lines {
    let text = extract_text_from_jsonl(&line);
    trainer.update_from_samples(std::iter::once(text.as_str()));
}

let byte_vocab: ByteMapVocab<T> = Default::default();
let vocab = trainer.train(byte_vocab)?;
```

### 3. 导出 HuggingFace `tokenizer.json`

`wordchipper` 不提供 HuggingFace 格式导出，需手动构建。关键步骤：

#### 3.1 构建 vocab 映射

使用 `vocab.unified_dictionary()` 获取 `{token_id -> Vec<u8>}` 映射，然后反转并转换为 `{token_string -> token_id}`。

对于 ByteLevel 编码，需要将字节序列转换为 HuggingFace 的 ByteLevel 字符串表示：
- 可打印 ASCII 字符（0x21-0x7E）直接映射为自身
- 其他字节映射为 Unicode 码点 `U+0100 + byte_value`（如 0x00 → Ā, 0x20 → Ġ）

```rust
fn byte_to_bytelevel_char(b: u8) -> char {
    if (0x21..=0x7E).contains(&b) {
        b as char
    } else {
        char::from_u32(0x100 + b as u32).unwrap()
    }
}
```

#### 3.2 构建 merges 列表

从 `vocab.pair_vocab().pair_map()` 获取所有合并对 `{(left_id, right_id) -> merged_id}`，然后按 `merged_id` 排序以还原训练顺序，再通过 `unified_dictionary()` 将 ID 映射回 ByteLevel 字符串。

```rust
let dict = vocab.unified_dictionary(); // {token_id -> Vec<u8>}
let mut merges: Vec<_> = vocab.pair_vocab().pair_map().iter().collect();
merges.sort_by_key(|(_, merged_id)| *merged_id);

let merge_list: Vec<[String; 2]> = merges.iter().map(|((left, right), _)| {
    let left_str = bytes_to_bytelevel_string(dict.get(left).unwrap());
    let right_str = bytes_to_bytelevel_string(dict.get(right).unwrap());
    [left_str, right_str]
}).collect();
```

#### 3.3 完整 JSON 结构

```json
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [ ...3 个特殊 token... ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": true
  },
  "post_processor": null,
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": { "token_string": token_id, ... },
    "merges": [ ["left", "right"], ... ]
  }
}
```

### 4. 导出 `tokenizer_config.json`

完全复刻 Python 脚本中的 config 字典，使用 `serde_json` 序列化。内容与原文件一致，包含 `chat_template`。

### 5. 输出路径

- `model/tokenizer.json`
- `model/tokenizer_config.json`

路径相对于项目根目录 `F:\MachineLearn\minimind\`，即 `../model/` 相对于 `rust-tokenizer/`。

## 边界条件与异常处理

1. **JSONL 文件不存在**：返回明确错误信息
2. **JSONL 行解析失败**：跳过该行并记录警告
3. **训练后特殊 token ID 不匹配**：断言失败并报错
4. **ByteLevel 字符编码**：确保字节到 Unicode 映射与 HuggingFace 一致（U+0100 偏移）
5. **merges 排序**：由于 `HashMap` 无序，必须按 `merged_id` 排序还原训练顺序

## 预期结果

- 运行 `cargo run --manifest-path rust-tokenizer/Cargo.toml` 后，在 `model/` 目录下生成与 Python 版本格式兼容的 `tokenizer.json` 和 `tokenizer_config.json`
- `tokenizer.json` 包含 6400 个 token 的词汇表和对应的 BPE merges
- `tokenizer_config.json` 包含完整的 HuggingFace transformers 配置
- 生成的文件可被 `transformers.AutoTokenizer.from_pretrained()` 正确加载
