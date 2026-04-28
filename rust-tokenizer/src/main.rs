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

    // 构建 vocab 对象
    let vocab_json: BTreeMap<&str, u32> = vocab_map
        .iter()
        .map(|(k, &v)| (k.as_str(), v))
        .collect();

    // 构建 merges 数组
    let merges_json: Vec<Vec<String>> = merge_list
        .iter()
        .map(|pair| vec![pair[0].clone(), pair[1].clone()])
        .collect();

    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
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
            "vocab": vocab_json,
            "merges": merges_json
        }
    });

    let mut file = fs::File::create(output_path)
        .with_context(|| format!("无法创建文件: {}", output_path.display()))?;
    let json_str = serde_json::to_string_pretty(&tokenizer_json)?;
    file.write_all(json_str.as_bytes())?;

    info!("tokenizer.json 已保存到 {}", output_path.display());
    Ok(())
}

/// 构建并保存 HuggingFace 格式的 tokenizer_config.json
fn save_tokenizer_config(output_path: &Path) -> Result<()> {
    // chat_template 与 Python 版本完全一致
    let chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within  specials XML tags:\\n\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n<|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith(' specials ') and message.content.endswith(' specials ')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n    special             {%- endif %}\n                {{- '\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n specials \\n' }}\n        {{- content }}\n        {{- '\\n specials ' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- ' specials \\n\\n specials \\n\\n' }}\n    {%- endif %}\n{%- endif %}";

    let config = json!({
        "add_bos_token": false,
        "add_eos_token": false,
        "add_prefix_space": false,
        "added_tokens_decoder": {
            "0": {
                "content": "",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false,
                "special": true
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false,
                "special": true
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": false,
                "normalized": false,
                "rstrip": false,
                "single_word": false,
                "special": true
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": false,
        "eos_token": "<|im_end|>",
        "legacy": true,
        "model_max_length": 32768,
        "pad_token": "",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": false,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "",
        "chat_template": chat_template
    });

    let mut file = fs::File::create(output_path)
        .with_context(|| format!("无法创建文件: {}", output_path.display()))?;
    let json_str = serde_json::to_string_pretty(&config)?;
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
    save_tokenizer_config(&config_json_path)?;

    info!("分词器训练完成，文件已保存到 {}", model_dir.display());
    Ok(())
}
