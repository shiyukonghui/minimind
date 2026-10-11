# 下载 Qwen3.8-Flash-Next 中 PLE(N-gram 记忆表)相关 safetensors 分片与 tokenizer 文件
# 背景：新版 hf CLI 默认走 xet CAS 通道（cas-server.xethub.hf.com），国内镜像不代理该域名导致 401；
#       本脚本使用 huggingface_hub 纯 HTTP 下载（snapshot_download + HF_HUB_DISABLE_XET=1）绕开该问题。
# 用法：python scripts/download_ple_shards.py [--list]  （--list 仅打印将要下载的文件清单）
import os, sys, argparse

# 必须在 import huggingface_hub 之前设置，禁用 xet 走传统 HTTP 解析
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download, list_repo_files

REPO_ID = "Qwen/Qwen3.8-Flash-Next"
LOCAL_DIR = r"H:\Strata-data\models\Qwen3.8-Flash-Next-BF16"

# 经权重索引验证：PLE 的 128 个 ngram_embedding.shard 分布在 model-00005~00037 共 33 个分片，
# 其中 00005/00037 为混合片（同时含 PLE 门控小张量），其余 31 片为纯表分片（~3.2GB/片）。
# 全仓库 360GB 中其余 98 个分片是 125B 主体 + 视觉塔，不需要下载。
PLE_SHARDS = [f"model-{i:05d}-of-00131.safetensors" for i in range(5, 38)]
SMALL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja"]
ALLOW = set(PLE_SHARDS + SMALL_FILES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="仅列出清单不下载")
    args = ap.parse_args()

    files = set(list_repo_files(REPO_ID))
    missing = ALLOW - files
    if missing:
        print(f"[警告] 仓库中不存在: {sorted(missing)}")
    to_get = sorted(ALLOW & files)
    print(f"将下载 {len(to_get)} 个文件 (~104GB)，输出目录: {LOCAL_DIR}")
    for f in to_get:
        print("  -", f)
    if args.list:
        return

    # 逐个下载：单文件失败自动重试，避免并发抢带宽导致大文件停滞在 0MB
    from huggingface_hub import hf_hub_download
    for i, f in enumerate(to_get):
        for attempt in range(5):
            try:
                p = hf_hub_download(REPO_ID, f, local_dir=LOCAL_DIR)
                print(f"[{i+1}/{len(to_get)}] OK {f} {os.path.getsize(p)/1e9:.2f}GB")
                break
            except Exception as e:
                print(f"[{i+1}/{len(to_get)}] 重试{attempt+1} {f}: {e}")
    print("下载流程结束")


if __name__ == "__main__":
    main()
