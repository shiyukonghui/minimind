# Phase 8: 强化学习后训练

> 本阶段目标：理解 LLM 强化学习的统一框架，掌握 DPO/PPO/GRPO/SPO 的原理与实现

## 一、强化学习后训练概述

### 1.1 为什么需要强化学习？

SFT 后的模型虽然能对话，但缺少正反样例的激励——不知道什么回答是好的，什么是差的。强化学习后训练的目标是让模型更符合偏好。

### 1.2 RLHF vs RLAIF

| 类型 | 反馈来源 | 优点 | 缺点 |
|------|---------|------|------|
| RLHF | 人类标注偏好 | 更贴近真实人类偏好 | 成本高、效率低 |
| RLAIF | AI 模型/规则 | 自动化、可扩展 | 可能偏离人类偏好 |

二者本质相同：通过强化学习利用"反馈"优化模型行为。

### 1.3 PO 算法的统一视角

所有 Policy Optimization (PO) 算法都在优化同一个期望：

$$\mathcal{J}_{PO} = \mathbb{E}_{q \sim P(Q), o \sim \pi(O|q)} \left[ \underbrace{f(r_t)}_{\text{策略项}} \cdot \underbrace{g(A_t)}_{\text{优势项}} - \underbrace{h(\text{KL}_t)}_{\text{正则项}} \right]$$

三个核心组件：

| 组件 | 含义 | 作用 |
|------|------|------|
| 策略项 $f(r_t)$ | 如何使用概率比 | 告诉模型新旧策略偏差有多大 |
| 优势项 $g(A_t)$ | 如何计算优势 | 衡量某个动作相比基线有多好 |
| 正则项 $h(\text{KL}_t)$ | 如何约束变化 | 防止策略偏离太远 |

**不同算法只是对这三个组件的不同实例化！**

## 二、DPO — 直接偏好优化

### 2.1 原理

DPO（Direct Preference Optimization）从 PPO 的 KL 约束目标推导出对偏好对的解析训练目标，直接最大化"chosen 优于 rejected"的对数几率。

**DPO 损失**：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$$

| 组件 | DPO 的设计 |
|------|-----------|
| 策略项 | $\log r_w - \log r_l$（对比 chosen vs rejected 的概率比） |
| 优势项 | 隐式（通过偏好对比，无需显式计算） |
| 正则项 | 隐含在 $\beta$ 中（控制偏离参考模型程度） |

**DPO 的特点**：
- Off-Policy：使用静态偏好数据集，可反复训练
- 只需 Actor + Ref 两个模型，显存占用低
- 无需训练 Reward Model
- 收敛稳定，实现简单

### 2.2 代码实现

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 按序列长度归一化 log_probs
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将 chosen 和 rejected 分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref = ref_log_probs[:batch_size // 2]
    reject_ref = ref_log_probs[batch_size // 2:]
    chosen_policy = policy_log_probs[:batch_size // 2]
    reject_policy = policy_log_probs[batch_size // 2:]

    # 计算 DPO 损失
    pi_logratios = chosen_policy - reject_policy
    ref_logratios = chosen_ref - reject_ref
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

### 2.3 启动训练

```bash
cd trainer
python train_dpo.py
```

关键参数：
- `--beta 0.1`：DPO 的 β 参数，控制偏离参考模型的程度
- `--learning_rate 4e-8`：极小的学习率，避免遗忘
- `--from_weight full_sft`：基于 SFT 模型训练

## 三、PPO — 近端策略优化

### 3.1 原理

PPO（Proximal Policy Optimization）是经典的强化学习算法，需要 Actor + Critic + Ref + Old Actor 四个模型。

**PPO 损失**：

$$\mathcal{L}_{PPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)\right] + \beta \cdot \mathbb{E}[\text{KL}]$$

| 组件 | PPO 的设计 |
|------|-----------|
| 策略项 | $\min(r_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon))$（裁剪概率比） |
| 优势项 | $R - V(s)$（通过 Critic 网络估计价值函数） |
| 正则项 | $\beta \cdot \mathbb{E}[\text{KL}]$（全局 KL 散度约束） |

**PPO 的特点**：
- On-Policy：必须用当前策略实时采样
- 需要 Actor + Critic 双网络联合优化
- 显存占用约为单网络方法的 1.5-2 倍
- 收敛较慢，但理论基础扎实

### 3.2 Critic 模型

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换 lm_head 为价值估计头
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```

### 3.3 奖励计算

```python
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    rewards = torch.zeros(len(responses), device=args.device)

    # 1. 格式奖励（推理模型专用）
    if args.reasoning == 1:
        # 检查思考标签格式
        pattern = r"^思考过程.*?最终回答$"
        format_rewards = [0.5 if re.match(pattern, r, re.S) else 0.0 for r in responses]
        rewards += torch.tensor(format_rewards, device=args.device)

    # 2. Reward Model 评分
    with torch.no_grad():
        for prompt, response in zip(prompts, responses):
            score = reward_model.get_score(reward_tokenizer, messages + [{"role": "assistant", "content": response}])
            score = max(min(score, 3.0), -3.0)  # 裁剪到 [-3, 3]
            reward_model_scores.append(score)
    rewards += torch.tensor(reward_model_scores, device=args.device)

    return rewards
```

### 3.4 启动训练

```bash
cd trainer
python train_ppo.py
```

需要先下载 Reward Model（InternLM2-1.8B-Reward）到项目同级目录。

## 四、GRPO — 分组相对策略优化

### 4.1 原理

GRPO（Group Relative Policy Optimization）来自 DeepSeekMath 论文，核心创新是"分组相对价值估计"。

**GRPO 损失**：

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[r_t \cdot A_t - \beta \cdot \text{KL}_t\right]$$

| 组件 | GRPO 的设计 |
|------|-----------|
| 策略项 | $r_t$（直接使用概率比，无 clip） |
| 优势项 | $\frac{R - \mu_{group}}{\sigma_{group}}$（组内归一化，消除 Critic） |
| 正则项 | $\beta \cdot \text{KL}_t$（token 级 KL 散度） |

**GRPO 的核心**：对同一个问题生成 N 个回答，用组内平均奖励作为 baseline：

```
问题: "1+1=?"
回答1: "2"     → reward = 2.5
回答2: "3"     → reward = -1.0
回答3: "等于2" → reward = 2.0
回答4: "不知道" → reward = -2.0

组内均值 = 0.375, 标准差 = 2.19

优势:
回答1: (2.5 - 0.375) / 2.19 = +0.97  → 鼓励
回答2: (-1.0 - 0.375) / 2.19 = -0.63  → 抑制
回答3: (2.0 - 0.375) / 2.19 = +0.74   → 鼓励
回答4: (-2.0 - 0.375) / 2.19 = -1.08  → 抑制
```

**GRPO 的优势**：不需要 Critic 网络，只需 Actor + Ref 两个模型。

**GRPO 的问题**：退化组——如果所有回答质量都差不多，学习信号接近零。

### 4.2 启动训练

```bash
cd trainer
python train_grpo.py
```

关键参数：
- `--num_generations 8`：每个 prompt 生成的样本数
- `--beta 0.02`：KL 惩罚系数

## 五、SPO — 单流策略优化

### 5.1 原理

SPO（Single-stream Policy Optimization）针对 GRPO 的退化组问题改进，回到"1 个输入，1 个输出，就是 1 个训练样本"的设计。

| 组件 | SPO 的设计 |
|------|-----------|
| 策略项 | $\log \pi_\theta(a_t|s)$（直接使用 log 概率） |
| 优势项 | $R - B_t^{adaptive}$（自适应 baseline，Beta 分布动态跟踪） |
| 正则项 | $\beta \cdot \text{KL}_t$（token 级 KL + 动态 ρ 调整） |

> 注：SPO 是实验性前沿算法，MiniMind 的实现用于探索学习。

### 5.2 启动训练

```bash
cd trainer
python train_spo.py
```

## 六、算法对比总结

| 算法 | 策略项 | 优势项 | 正则项 | 优化模型数 | 特点 |
|------|--------|--------|--------|----------|------|
| DPO | $\log r_w - \log r_l$ | 隐式 | 隐含在 β | 2 | 简单稳定，Off-Policy |
| PPO | $\min(r, \text{clip}(r))$ | $R - V(s)$ | $\beta \cdot \mathbb{E}[\text{KL}]$ | 4 | 经典基线，On-Policy |
| GRPO | $r$ | $\frac{R - \mu}{\sigma}$ | $\beta \cdot \text{KL}_t$ | 2 | 无需 Critic，On-Policy |
| SPO | $\log \pi_\theta$ | $R - B_t^{adaptive}$ | $\beta \cdot \text{KL}_t$ | 2 | 无退化组，实验性 |

**训练曲线对比**：
- PPO：reward 提升缓慢（双网络联合优化复杂）
- GRPO：reward 上升更稳定，收敛上限更高
- SPO：reward 波动与 PPO 接近

## 七、Reward Model 准备

### 7.1 下载奖励模型

使用 InternLM2-1.8B-Reward：

```bash
# ModelScope
git clone https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward

# HuggingFace
git clone https://huggingface.co/internlm/internlm2-1_8b-reward
```

### 7.2 目录结构

```
project/
├── minimind/                    # MiniMind 项目
└── internlm2-1_8b-reward/       # 奖励模型（与 minimind 同级）
```

### 7.3 奖励稀疏问题

对于 MiniMind 这种超小模型，在通用任务上会遇到严重的奖励稀疏问题：

- **现象**：模型生成的回答几乎全部错误，奖励分数 $r(x,y) \approx 0$
- **后果**：优势函数 $A(x,y) \approx 0$，策略梯度信号消失

**缓解方案**：
- 使用 Model-based 的连续性奖励信号（而非二元 0/1）
- 监控奖励分数的方差，若持续接近 0 则需调整数据或奖励机制
- 限制在模型能力边界内的任务

## 八、动手练习

### 练习 1：DPO 训练

```bash
cd trainer
python train_dpo.py --epochs 1 --batch_size 4
```

### 练习 2：对比 DPO 前后的模型

```bash
# DPO 前
python eval_llm.py --weight full_sft

# DPO 后
python eval_llm.py --weight dpo
```

### 练习 3：GRPO 训练（需要 Reward Model）

```bash
cd trainer
python train_grpo.py --num_generations 4
```

## 九、下一阶段预告

下一阶段 [Phase 9: 推理模型与部署](./09_Phase9_推理模型与部署.md)，我们将学习：
- 推理模型的蒸馏训练
- OpenAI API 兼容服务
- 第三方推理框架集成
