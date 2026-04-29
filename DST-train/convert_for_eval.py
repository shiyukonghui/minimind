"""
模型格式转换脚本
将 DST-train/model/ 下的 PyTorch 权重转换为 HuggingFace 格式，供 eval_benchmark.py 使用
支持转换：baseline / dst_phase1 / dst_recovered 三种模型
"""
import os
import sys

# 确保可以导入项目根目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import warnings
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


# 模型类型到权重文件名的映射
MODEL_TYPES = {
    'baseline': 'baseline_{hidden_size}.pth',
    'dst_phase1': 'dst_phase1_{hidden_size}.pth',
    'dst_pruned': 'dst_pruned_{hidden_size}_sp{sparsity}.pth',
    'dst_recovered': 'dst_recovered_{hidden_size}_sp{sparsity}.pth',
}


def convert_torch_to_hf(torch_path, hf_output_dir, hidden_size=512, num_hidden_layers=8,
                        use_moe=False, dtype=torch.float16, tokenizer_path='./model/'):
    """将 PyTorch 权重转换为 HuggingFace 格式

    Args:
        torch_path: PyTorch 权重文件路径
        hf_output_dir: HuggingFace 格式输出目录
        hidden_size: 隐藏层维度
        num_hidden_layers: 层数
        use_moe: 是否使用MoE
        dtype: 保存精度
        tokenizer_path: 分词器路径
    """
    print(f'正在转换: {torch_path} → {hf_output_dir}')

    # 构建模型
    lm_config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe
    )

    # 注册AutoClass以支持HuggingFace自动加载
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = MiniMindForCausalLM(lm_config)

    # 加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  缺失的key: {len(missing)} 个')
    if unexpected:
        print(f'  多余的key: {len(unexpected)} 个')

    # 转换精度
    model = model.to(dtype)

    # 打印模型参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  模型参数: {model_params / 1e6:.2f}M')

    # 保存为HuggingFace格式
    os.makedirs(hf_output_dir, exist_ok=True)
    model.save_pretrained(hf_output_dir, safe_serialization=False)

    # 复制分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(hf_output_dir)

    print(f'  HuggingFace格式模型已保存至: {hf_output_dir}')


def main():
    parser = argparse.ArgumentParser(description="DST模型格式转换：PyTorch → HuggingFace")
    parser.add_argument("--model_dir", type=str, default="./model", help="PyTorch权重所在目录")
    parser.add_argument("--model_type", type=str, default="all",
                        choices=["all", "baseline", "dst_phase1", "dst_pruned", "dst_recovered"],
                        help="要转换的模型类型")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--sparsity', default=20, type=int, help="剪枝稀疏度百分比(如20表示20%%)")
    parser.add_argument("--dtype", type=str, default="float16", help="保存精度(float16/bfloat16)")
    parser.add_argument("--tokenizer_path", type=str, default="./model/", help="分词器路径")
    args = parser.parse_args()

    save_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    moe_suffix = '_moe' if args.use_moe else ''

    # 确定要转换的模型列表
    if args.model_type == "all":
        types_to_convert = ["baseline", "dst_recovered"]
    else:
        types_to_convert = [args.model_type]

    for model_type in types_to_convert:
        # 构建权重文件路径
        template = MODEL_TYPES[model_type]
        weight_name = template.format(
            hidden_size=args.hidden_size,
            sparsity=args.sparsity
        ) + moe_suffix + '.pth' if moe_suffix else template.format(
            hidden_size=args.hidden_size,
            sparsity=args.sparsity
        ) + '.pth'

        # 更精确的文件名构建
        if model_type == 'baseline':
            weight_name = f'baseline_{args.hidden_size}{moe_suffix}.pth'
            hf_dir = f'{args.model_dir}/baseline_hf'
        elif model_type == 'dst_phase1':
            weight_name = f'dst_phase1_{args.hidden_size}{moe_suffix}.pth'
            hf_dir = f'{args.model_dir}/dst_phase1_hf'
        elif model_type == 'dst_pruned':
            weight_name = f'dst_pruned_{args.hidden_size}_sp{args.sparsity}{moe_suffix}.pth'
            hf_dir = f'{args.model_dir}/dst_pruned_sp{args.sparsity}_hf'
        elif model_type == 'dst_recovered':
            weight_name = f'dst_recovered_{args.hidden_size}_sp{args.sparsity}{moe_suffix}.pth'
            hf_dir = f'{args.model_dir}/dst_hf'

        torch_path = f'{args.model_dir}/{weight_name}'

        if not os.path.exists(torch_path):
            print(f'跳过 {model_type}: 权重文件不存在 ({torch_path})')
            continue

        convert_torch_to_hf(
            torch_path=torch_path,
            hf_output_dir=hf_dir,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            dtype=save_dtype,
            tokenizer_path=args.tokenizer_path
        )

    print('\n转换完成！')


if __name__ == '__main__':
    main()
