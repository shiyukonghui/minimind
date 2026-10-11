# DeepSeek-V4.1-Flash 教师客户端（设计文档 §3.7 RSI 阶段1，路线图任务⑧前置）
#
# 闭源 API 模型只能做序列级蒸馏：教师生成高质量回答 -> 学生 SFT/LoRA 学习。
# 本客户端用标准库 urllib 实现 OpenAI 兼容 chat/completions 调用，
# 不依赖 openai SDK 版本（本机 openai 0.27.8 与新版 API 不兼容）。
#
# 特性：
#   - API key 从环境变量 DEEPSEEK_API_KEY 读取（无 key 时 dry_run 模式走本地占位）；
#   - 磁盘缓存(jsonl)：按 messages 哈希去重，断点续跑不重复消耗额度；
#   - 指数退避重试 + 请求间隔限速。
import os, sys, json, time, hashlib
import urllib.request
import urllib.error

API_BASE = os.environ.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
API_KEY_ENV = 'DEEPSEEK_API_KEY'
MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4.1-flash')


def _hash(messages) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


class TeacherClient:
    def __init__(self, cache_path: str = None, dry_run: bool = None,
                 temperature: float = 0.7, max_tokens: int = 1024, min_interval: float = 0.5):
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.cache[rec['hash']] = rec['response']
        # 未显式指定时：无 API key 自动 dry_run，便于无 key 环境验证全链路
        self.dry_run = (os.environ.get(API_KEY_ENV) is None) if dry_run is None else dry_run
        self.temperature, self.max_tokens = temperature, max_tokens
        self.min_interval = min_interval
        self._last_call = 0.0

    def _post(self, payload: dict, retries: int = 4) -> str:
        url = f'{API_BASE}/chat/completions'
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.environ.get(API_KEY_ENV)}',
        })
        for i in range(retries):
            try:
                # 限速：两次真实调用间隔不小于 min_interval
                wait = self.min_interval - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode())
                self._last_call = time.time()
                return body['choices'][0]['message']['content']
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as e:
                if i == retries - 1:
                    raise RuntimeError(f'教师调用失败: {e}') from e
                time.sleep(2 ** i)     # 指数退避

    def chat(self, messages) -> str:
        """messages: [{role, content}, ...] -> 教师回复文本。命中缓存则直接返回。"""
        key = _hash(messages)
        if key in self.cache:
            return self.cache[key]
        if self.dry_run:
            # dry_run：占位回复，仅用于验证数据管道，不产生真实蒸馏价值
            reply = f'[DRY_RUN] 这是对 "{messages[-1]["content"][:40]}" 的占位回答'
        else:
            reply = self._post({
                'model': MODEL,
                'messages': messages,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
            })
        if self.cache_path:
            self.cache[key] = reply
            with open(self.cache_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'hash': key, 'messages': messages, 'response': reply},
                                   ensure_ascii=False) + '\n')
        return reply
