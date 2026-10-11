# ⑥ 字节化前置产物冒烟测试：ByteTokenizer 往返 / 字节数据集 / byteinit 模型加载前向
# 用法：python scripts/test_byte_pipeline.py
import sys
import torch

sys.path.insert(0, r'F:\MachineLearn\minimind')
from model.byte_tokenizer import ByteTokenizer, BYTE_VOCAB_SIZE
from dataset.lm_dataset import ByteSFTDataset, BytePretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

DATA_DIR = r'F:\MachineLearn\minimind\dataset'
BYTE_CKPT = r'F:\MachineLearn\minimind\checkpoints\pretrain_512_byteinit.pth'

def test_tokenizer():
    tk = ByteTokenizer()
    texts = ['hello world', '深度学习字节化', '混合 mixed 文本\n换行\t制表', 'emoji 🎉 测试']
    for t in texts:
        ids = tk.encode(t)
        assert all(0 <= i < 256 for i in ids), t
        assert tk.decode(ids) == t, (t, tk.decode(ids))
    # 特殊 token 往返
    ids = tk.encode('hi', add_special_tokens=True)
    assert ids[0] == tk.bos_token_id and ids[-1] == tk.eos_token_id
    # chat 模板
    s = tk.apply_chat_template([{'role': 'user', 'content': 'a'}], add_generation_prompt=True)
    assert tk.bos_token in s and 'assistant' in s
    print(f'[1] ByteTokenizer 往返 ✓  vocab={BYTE_VOCAB_SIZE}，中文/emoji/控制符无损')

def test_datasets():
    tk = ByteTokenizer()
    ds = ByteSFTDataset(f'{DATA_DIR}/sft_t2t_mini.jsonl', tk, max_length=256)
    X, Y, M = ds[0]
    assert X.shape == Y.shape == M.shape == (255,)
    assert X.max() < BYTE_VOCAB_SIZE and Y.max() < BYTE_VOCAB_SIZE
    assert M.sum() > 0, 'assistant 段应有监督位'
    # 解码验证：监督位应落在 assistant 内容上
    sup_text = tk.decode(Y[M == 1].tolist())
    assert len(sup_text) > 0
    dp = BytePretrainDataset(f'{DATA_DIR}/pretrain_t2t_mini.jsonl', tk, max_length=128)
    X2, Y2, M2 = dp[0]
    assert X2.shape == (127,) and M2.dtype == torch.long
    print(f'[2] 字节数据集 ✓  SFT 样本形状{tuple(X.shape)} 监督位={int(M.sum())}，'
          f'pretrain 形状{tuple(X2.shape)}')

def test_byte_model():
    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                         num_key_value_heads=2, vocab_size=BYTE_VOCAB_SIZE)
    m = MiniMindForCausalLM(cfg)
    sd = torch.load(BYTE_CKPT, map_location='cpu')['model_state']
    m.load_state_dict(sd, strict=True)
    tk = ByteTokenizer()
    ids = torch.tensor([tk.encode('字节化模型前向测试')], dtype=torch.long)
    with torch.no_grad():
        out = m(ids)
    assert out.logits.shape == (1, ids.shape[1], BYTE_VOCAB_SIZE)
    assert torch.isfinite(out.logits).all()
    # 字节嵌入不应全零(ALP 折叠生效)
    e = sd['model.embed_tokens.weight']
    assert e[:256].abs().sum() > 0 and e.std() > 1e-4
    print(f'[3] 字节模型 ✓  byteinit strict 加载，logits{tuple(out.logits.shape)}，'
          f'embed std={e[:256].std():.4f}')

if __name__ == '__main__':
    test_tokenizer()
    test_datasets()
    test_byte_model()
    print('\n全部通过')
