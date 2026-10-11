# 字节级 Tokenizer（设计文档 §3.6 字节化路线，路线图任务⑥前置）
#
# 词表 = 256 个 UTF-8 字节 + 4 个特殊 token（PAD/BOS/EOS/UNK），与 subword 词表解耦，
# 使 Engram/PLE 的 N-gram 哈希键空间跨模型通用。接口对齐 HF tokenizer 常用子集，
# 可直接喂给 dataset 中的字节版数据集类。
import torch

BYTE_PAD = 256      # 特殊 token：pad
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_UNK = 259      # 字节化后不存在未知字节，保留占位与 HF 接口兼容

BYTE_VOCAB_SIZE = 260


class ByteTokenizer:
    """UTF-8 字节 tokenizer：文本 <-> 字节 id 序列（0-255），特殊 id 256-259。"""

    pad_token_id = BYTE_PAD
    bos_token_id = BYTE_BOS
    eos_token_id = BYTE_EOS
    unk_token_id = BYTE_UNK
    vocab_size = BYTE_VOCAB_SIZE
    pad_token = '<|pad|>'
    bos_token = '<|bos|>'
    eos_token = '<|eos|>'

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        ids = list(text.encode('utf-8'))
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids) -> str:
        """字节 id -> 文本；特殊 token 跳过，非法字节序列按 replace 解码。"""
        if torch.is_tensor(ids):
            ids = ids.tolist()
        raw = bytes(b for b in ids if b < 256)
        return raw.decode('utf-8', errors='replace')

    def __call__(self, text: str, max_length: int = None, truncation: bool = False,
                 add_special_tokens: bool = False, return_tensors: str = None):
        ids = self.encode(text, add_special_tokens)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        out = {'input_ids': torch.tensor([ids], dtype=torch.long)}
        return out if return_tensors == 'pt' else {'input_ids': ids}

    def apply_chat_template(self, messages, add_generation_prompt: bool = False,
                            tokenize: bool = False, tools=None, **kwargs):
        """极简 chat 模板：每条消息渲染为 <|bos|>{role}\n{content}<|eos|>\n。

        字节 tokenizer 无 HF 模板文件，这里提供文本级渲染，tokenize=True 时返回字节 id。
        """
        parts = []
        for m in messages:
            parts.append(f'{self.bos_token}{m["role"]}\n{m["content"]}{self.eos_token}\n')
        if add_generation_prompt:
            parts.append(f'{self.bos_token}assistant\n')
        text = ''.join(parts)
        return self.encode(text) if tokenize else text
