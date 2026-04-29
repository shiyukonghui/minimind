use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use log::info;
use serde_json::{json, Value};
use wordchipper::pretrained::openai::OA_GPT2_PATTERN_SLOW;
use wordchipper::vocab::ByteMapVocab;
use wordchipper_training::{BPETRainerOptions, BPETrainer};

/// 特殊 token 定义：(token字符串, 在HuggingFace格式中的目标ID)
const SPECIAL_TOKENS: &[(&str, u32)] = &[
    ("", 0),
    ("<|im_start|>", 1),
    ("<|im_end|>", 2),
];

/// 目标词汇表大小
const VOCAB_SIZE: usize = 6400;

/// BPE 训练的词汇表大小（需减去特殊 token 的数量）
/// wordchipper 训练时 byte vocab 占用 ID 0-255，训练后我们需要在导出时做 ID 重映射
const TRAIN_VOCAB_SIZE: usize = VOCAB_SIZE - SPECIAL_TOKENS.len();

/// ByteLevel 编码：将字节转换为 HuggingFace ByteLevel 字符表示
/// 可打印 ASCII 字符 (0x21-0x7E) 直接映射为自身，其他字节映射为 U+0100 + byte_value
fn byte_to_bytelevel_char(b: u8) -> char {
    if (0x21..=0x7E).contains(&b) {
        b as char
    } else {
        char::from_u32(0x100 + b as u32).unwrap()
    }
}

/// 将字节切片转换为 HuggingFace ByteLevel 字符串表示
fn bytes_to_bytelevel_string(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| byte_to_bytelevel_char(b)).collect()
}

/// 从 JSONL 文件中逐行读取文本数据
fn read_texts_from_jsonl(file_path: &Path) -> Result<Vec<String>> {
    let file = fs::File::open(file_path)
        .with_context(|| format!("无法打开JSONL文件: {}", file_path.display()))?;
    let reader = BufReader::new(file);
    let mut texts = Vec::new();
    let mut line_num = 0;

    for line in reader.lines() {
        line_num += 1;
        let line = line.with_context(|| format!("读取第 {} 行失败", line_num))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(data) => {
                if let Some(text) = data.get("text").and_then(|t| t.as_str()) {
                    texts.push(text.to_string());
                } else {
                    info!("第 {} 行缺少 'text' 字段，跳过", line_num);
                }
            }
            Err(e) => {
                info!("第 {} 行 JSON 解析失败: {}，跳过", line_num, e);
            }
        }
    }

    info!("从 {} 读取了 {} 条文本", file_path.display(), texts.len());
    Ok(texts)
}

/// 从 span_vocab 推导 BPE 规范合并列表
/// 对每个非字节 token，找到其唯一的规范分解（BPE 训练时产生该 token 的合并操作）
fn extract_canonical_merges(
    span_vocab: &wordchipper::vocab::SpanMapVocab<u32>,
) -> Vec<((u32, u32), u32)> {
    // 构建 bytes -> token_id 的查找表
    let lookup: BTreeMap<Vec<u8>, u32> = span_vocab
        .iter()
        .map(|(bytes, &token_id)| (bytes.to_vec(), token_id))
        .collect();

    // 构建 token_id -> bytes 的反向映射
    let token_to_bytes: BTreeMap<u32, Vec<u8>> = span_vocab
        .iter()
        .map(|(bytes, &token_id)| (token_id, bytes.to_vec()))
        .collect();

    let mut merges: Vec<((u32, u32), u32)> = Vec::new();

    // 对每个非字节 token（ID > 255），找到其规范分解
    for (&token_id, bytes) in &token_to_bytes {
        if token_id < 256 || bytes.len() < 2 {
            continue;
        }

        // 遍历所有二分位置，找到两个子 token 都在 vocab 中的分解
        // BPE 规范分解是唯一的：选择使得两个子 token ID 都小于当前 token ID 的分解
        // 如果有多个，选择子 token ID 之和最大的（BPE 自底向上构建，最后合并的子 token 更大）
        let mut best_merge: Option<((u32, u32), u32)> = None;
        let mut best_sum: u32 = 0;

        for p in 1..bytes.len() {
            let pre = &bytes[..p];
            let post = &bytes[p..];

            if let (Some(&left_id), Some(&right_id)) = (lookup.get(pre), lookup.get(post)) {
                // BPE 规范分解：子 token ID 必须都小于当前 token ID
                if left_id < token_id && right_id < token_id {
                    let sum = left_id + right_id;
                    if sum > best_sum {
                        best_sum = sum;
                        best_merge = Some(((left_id, right_id), token_id));
                    }
                }
            }
        }

        if let Some(merge) = best_merge {
            merges.push(merge);
        }
    }

    // 按合并后的 token ID 排序（即 BPE 训练顺序）
    merges.sort_by_key(|(_, merged_id)| *merged_id);
    merges
}

/// 训练 BPE 分词器，返回 (词汇表映射, 合并规则列表)
fn train_bpe(texts: &[String]) -> Result<(BTreeMap<String, u32>, Vec<[String; 2]>)> {
    // 初始化训练器（使用 GPT-2 的 ByteLevel 等价正则）
    let mut trainer: BPETrainer =
        BPETRainerOptions::new(OA_GPT2_PATTERN_SLOW, TRAIN_VOCAB_SIZE).init();

    // 分批喂入训练数据
    info!("开始训练 BPE 分词器...");
    trainer.update_from_samples(texts.iter().map(|s| s.as_str()));

    // 训练得到 UnifiedTokenVocab
    let byte_vocab: ByteMapVocab<u32> = Default::default();
    let vocab = trainer.train(byte_vocab).context("BPE 训练失败")?;

    info!("训练完成，开始构建词汇表映射...");

    // 获取统一的 token -> bytes 映射
    let dict = vocab.unified_dictionary();

    // 从 span_vocab 推导 BPE 规范合并列表
    let merges = extract_canonical_merges(vocab.span_vocab());
    info!("规范合并规则数量: {}", merges.len());

    // 构建词汇表：将 wordchipper 的 ID 映射到 HuggingFace 格式的 ID
    // wordchipper 的 byte vocab 占用 ID 0-255，BPE 合并的 token 从 256 开始
    // HuggingFace 格式中：特殊 token 占用 ID 0-2，byte vocab 占用 ID 3-258，BPE 合并从 259 开始
    let offset = SPECIAL_TOKENS.len() as u32;

    let mut vocab_map: BTreeMap<String, u32> = BTreeMap::new();

    // 添加特殊 token
    for &(token_str, token_id) in SPECIAL_TOKENS {
        vocab_map.insert(token_str.to_string(), token_id);
    }

    // 添加 byte vocab + BPE 合并 token
    for (token_id, bytes) in dict.iter() {
        let hf_id = token_id + offset;
        let token_str = bytes_to_bytelevel_string(bytes);
        vocab_map.insert(token_str, hf_id);
    }

    info!("词汇表大小: {}", vocab_map.len());

    // 构建合并规则字符串列表
    let merge_list: Vec<[String; 2]> = merges
        .iter()
        .map(|((left_id, right_id), _)| {
            let left_str = dict
                .get(left_id)
                .map(|b| bytes_to_bytelevel_string(b))
                .unwrap_or_default();
            let right_str = dict
                .get(right_id)
                .map(|b| bytes_to_bytelevel_string(b))
                .unwrap_or_default();
            [left_str, right_str]
        })
        .collect();

    Ok((vocab_map, merge_list))
}

/// 构建并保存 HuggingFace 格式的 tokenizer.json
fn save_tokenizer_json(
    output_path: &Path,
    vocab_map: &BTreeMap<String, u32>,
    merge_list: &[[String; 2]],
) -> Result<()> {
    // 构建 added_tokens 数组
    let added_tokens: Vec<Value> = SPECIAL_TOKENS
        .iter()
        .map(|&(content, id)| {
            json!({
                "id": id,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            })
        })
        .collect();

    // 构建 vocab 对象（保持插入顺序）
    let vocab_json: serde_json::Map<String, Value> = vocab_map
        .iter()
        .map(|(k, &v)| (k.clone(), json!(v)))
        .collect();

    // 构建 merges 数组
    let merges_json: Vec<Vec<String>> = merge_list
        .iter()
        .map(|pair| vec![pair[0].clone(), pair[1].clone()])
        .collect();

    // 手动构建 JSON 以控制字段顺序
    let mut json_obj = serde_json::Map::new();
    json_obj.insert("version".into(), json!("1.0"));
    json_obj.insert("truncation".into(), json!(null));
    json_obj.insert("padding".into(), json!(null));
    json_obj.insert("added_tokens".into(), json!(added_tokens));
    json_obj.insert("normalizer".into(), json!(null));

    let mut pre_tokenizer = serde_json::Map::new();
    pre_tokenizer.insert("type".into(), json!("ByteLevel"));
    pre_tokenizer.insert("add_prefix_space".into(), json!(false));
    pre_tokenizer.insert("trim_offsets".into(), json!(true));
    pre_tokenizer.insert("use_regex".into(), json!(true));
    json_obj.insert("pre_tokenizer".into(), Value::Object(pre_tokenizer));

    json_obj.insert("post_processor".into(), json!(null));

    let mut decoder = serde_json::Map::new();
    decoder.insert("type".into(), json!("ByteLevel"));
    decoder.insert("add_prefix_space".into(), json!(true));
    decoder.insert("trim_offsets".into(), json!(true));
    decoder.insert("use_regex".into(), json!(true));
    json_obj.insert("decoder".into(), Value::Object(decoder));

    let mut model = serde_json::Map::new();
    model.insert("type".into(), json!("BPE"));
    model.insert("dropout".into(), json!(null));
    model.insert("unk_token".into(), json!(null));
    model.insert("continuing_subword_prefix".into(), json!(null));
    model.insert("end_of_word_suffix".into(), json!(null));
    model.insert("fuse_unk".into(), json!(false));
    model.insert("byte_fallback".into(), json!(false));
    model.insert("ignore_merges".into(), json!(false));
    model.insert("vocab".into(), Value::Object(vocab_json));
    model.insert("merges".into(), json!(merges_json));
    json_obj.insert("model".into(), Value::Object(model));

    let mut file = fs::File::create(output_path)
        .with_context(|| format!("无法创建文件: {}", output_path.display()))?;
    let json_str = serde_json::to_string_pretty(&Value::Object(json_obj))?;
    file.write_all(json_str.as_bytes())?;

    info!("tokenizer.json 已保存到 {}", output_path.display());
    Ok(())
}

/// 构建并保存 HuggingFace 格式的 tokenizer_config.json
/// 从 Python 脚本对应的源文件中读取 chat_template，避免在 Rust 代码中硬编码含 XML 标签的模板
fn save_tokenizer_config(output_path: &Path, model_dir: &Path) -> Result<()> {
    // 尝试从已有的 tokenizer_config.json 中读取 chat_template
    // 如果不存在，则使用空字符串作为默认值
    let existing_config_path = model_dir.join("tokenizer_config.json");
    let chat_template = if existing_config_path.exists() {
        let content = fs::read_to_string(&existing_config_path)?;
        let parsed: Value = serde_json::from_str(&content)?;
        parsed
            .get("chat_template")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    };

    let mut config = serde_json::Map::new();
    config.insert("add_bos_token".into(), json!(false));
    config.insert("add_eos_token".into(), json!(false));
    config.insert("add_prefix_space".into(), json!(false));

    // 构建 added_tokens_decoder
    let mut atd = serde_json::Map::new();
    for &(content_str, id) in SPECIAL_TOKENS {
        let mut token_obj = serde_json::Map::new();
        token_obj.insert("content".into(), json!(content_str));
        token_obj.insert("lstrip".into(), json!(false));
        token_obj.insert("normalized".into(), json!(false));
        token_obj.insert("rstrip".into(), json!(false));
        token_obj.insert("single_word".into(), json!(false));
        token_obj.insert("special".into(), json!(true));
        atd.insert(id.to_string(), Value::Object(token_obj));
    }
    config.insert("added_tokens_decoder".into(), Value::Object(atd));

    config.insert("additional_special_tokens".into(), json!([]));
    config.insert("bos_token".into(), json!("<|im_start|>"));
    config.insert("clean_up_tokenization_spaces".into(), json!(false));
    config.insert("eos_token".into(), json!("<|im_end|>"));
    config.insert("legacy".into(), json!(true));
    config.insert("model_max_length".into(), json!(32768));
    config.insert("pad_token".into(), json!(""));
    config.insert("sp_model_kwargs".into(), json!({}));
    config.insert("spaces_between_special_tokens".into(), json!(false));
    config.insert("tokenizer_class".into(), json!("PreTrainedTokenizerFast"));
    config.insert("unk_token".into(), json!(""));
    config.insert("chat_template".into(), json!(chat_template));

    let mut file = fs::File::create(output_path)
        .with_context(|| format!("无法创建文件: {}", output_path.display()))?;
    let json_str = serde_json::to_string_pretty(&Value::Object(config))?;
    file.write_all(json_str.as_bytes())?;

    info!("tokenizer_config.json 已保存到 {}", output_path.display());
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // 数据文件路径（相对于 rust-tokenizer 项目根目录）
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_path = project_dir.join("../dataset/pretrain_t2t_mini.jsonl");
    let model_dir = project_dir.join("../model");

    // 确保输出目录存在
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("无法创建目录: {}", model_dir.display()))?;

    // 1. 读取训练数据
    info!("读取训练数据: {}", data_path.display());
    let texts = read_texts_from_jsonl(&data_path)?;

    if texts.is_empty() {
        bail!("未读取到任何训练数据");
    }

    // 2. 训练 BPE 分词器
    let (vocab_map, merge_list) = train_bpe(&texts)?;

    // 3. 验证特殊 token ID
    for &(token_str, expected_id) in SPECIAL_TOKENS {
        let actual_id = vocab_map
            .get(token_str)
            .copied()
            .with_context(|| format!("特殊 token '{}' 不在词汇表中", token_str))?;
        if actual_id != expected_id {
            bail!(
                "特殊 token '{}' 的 ID 为 {}，期望为 {}",
                token_str,
                actual_id,
                expected_id
            );
        }
    }
    info!("特殊 token ID 验证通过");

    // 4. 保存 tokenizer.json
    let tokenizer_json_path = model_dir.join("tokenizer.json");
    save_tokenizer_json(&tokenizer_json_path, &vocab_map, &merge_list)?;

    // 5. 保存 tokenizer_config.json
    let config_json_path = model_dir.join("tokenizer_config.json");
    save_tokenizer_config(&config_json_path, &model_dir)?;

    info!("分词器训练完成，文件已保存到 {}", model_dir.display());
    Ok(())
}
