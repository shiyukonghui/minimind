"""
模型格式转换脚本
将 embedding-exp/model/ 下的 PyTorch 权重转换为 HuggingFace 格式，供 eval_embedding_exp.py 使用
支持转换方案 B/C1/C2 的模型（方案A已是HuggingFace格式，无需转换）
"""
import os
import sys

# 确保可以导入项目根目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import warnings
from transformers import AutoTokenizer
from model_embedding import MiniMindEmbeddingConfig, MiniMindEmbeddingForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch_to_hf(torch_path, hf_output_dir, embed_type='B',
                        hidden_size=512, num_hidden_layers=8, use_moe=False,
                        dtype=torch.float16, tokenizer_path='./model/'):
    """将 PyTorch 权重转换为 HuggingFace 格式

    Args:
        torch_path: PyTorch 权重文件路径
        hf_output_dir: HuggingFace 格式输出目录
        embed_type: 嵌入方案类型
        hidden_size: 隐藏层维度
        num_hidden_layers: 层数
        use_moe: 是否使用MoE
        dtype: 保存精度
        tokenizer_path: 分词器路径
    """
    print(f'正在转换: {torch_path} -> {hf_output_dir}')

    # 注册AutoClass以支持HuggingFace自动加载
    MiniMindEmbeddingConfig.register_for_auto_class()
    MiniMindEmbeddingForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    # 构建模型
    lm_config = MiniMindEmbeddingConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
        embed_type=embed_type
    )
    model = MiniMindEmbeddingForCausalLM(lm_config)

    # 加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  缺失的key: {len(missing)} 个')
        for k in missing[:5]:
            print(f'    - {k}')
        if len(missing) > 5:
            print(f'    ... 共{len(missing)}个')
    if unexpected:
        print(f'  多余的key: {len(unexpected)} 个')
        for k in unexpected[:5]:
            print(f'    - {k}')
        if len(unexpected) > 5:
            print(f'    ... 共{len(unexpected)}个')

    # 转换精度
    model = model.to(dtype)

    # 打印模型参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed_params = model.get_embed_param_count()
    print(f'  模型参数: {model_params / 1e6:.2f}M, 嵌入相关参数: {embed_params / 1e3:.1f}K')

    # 保存为HuggingFace格式
    os.makedirs(hf_output_dir, exist_ok=True)
    model.save_pretrained(hf_output_dir, safe_serialization=False)

    # 复制分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(hf_output_dir)

    print(f'  HuggingFace格式模型已保存至: {hf_output_dir}')


def main():
    parser = argparse.ArgumentParser(description="Embedding Exp: PyTorch -> HuggingFace")
    parser.add_argument("--model_dir", type=str, default="./model", help="PyTorch权重所在目录")
    parser.add_argument("--embed_type", type=str, default="all",
                        choices=["all", "B", "C1", "C2"],
                        help="要转换的嵌入方案")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument("--dtype", type=str, default="float16", help="保存精度(float16/bfloat16)")
    parser.add_argument("--tokenizer_path", type=str, default="./model/",
                        help="分词器路径（相对于项目根目录）")
    args = parser.parse_args()

    save_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    moe_suffix = '_moe' if args.use_moe else ''

    # 确定要转换的方案列表
    if args.embed_type == "all":
        types_to_convert = ["B", "C1", "C2"]
    else:
        types_to_convert = [args.embed_type]

    for embed_type in types_to_convert:
        weight_name = f'{embed_type.lower()}_{args.hidden_size}{moe_suffix}.pth'
        torch_path = f'{args.model_dir}/{weight_name}'
        hf_dir = f'{args.model_dir}/{embed_type.lower()}_hf'

        if not os.path.exists(torch_path):
            print(f'跳过方案 {embed_type}: 权重文件不存在 ({torch_path})')
            continue

        convert_torch_to_hf(
            torch_path=torch_path,
            hf_output_dir=hf_dir,
            embed_type=embed_type,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            dtype=save_dtype,
            tokenizer_path=args.tokenizer_path
        )

    print('\n转换完成!')


if __name__ == '__main__':
    main()
