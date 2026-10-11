# 验证从 HF 镜像下载的 Qwen tokenizer 是否完整可用（任务①）
from tokenizers import Tokenizer

tok = Tokenizer.from_file(r"H:\Strata-data\models\Qwen3.8-Flash-Next-BF16\tokenizer.json")
print("vocab_size:", tok.get_vocab_size())

ids = tok.encode("hello world 你好", add_special_tokens=False).ids
print("encode sample:", ids)

# EOS 特殊 token（哈希公式里用于重置 n-gram 窗口）；GGUF 元数据给出 eos=248044
for tid in (248044, 248056):
    print(f"id {tid} ->", repr(tok.decode([tid])))

# 逐 token 解码，检查是否存在单字节 token（决定 PLE 键空间能否用字节直接构造）
vocab = tok.get_vocab()
single_byte = [t for t in vocab if len(t.encode("utf-8")) == 1]
print("single_byte_token_count:", len(single_byte))