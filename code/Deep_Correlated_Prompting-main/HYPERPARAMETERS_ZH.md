# 反事实缺失模态模型：超参数说明

## 一、建议的起始配置

第一轮不要直接使用强对抗训练。建议从以下配置开始：

```text
learning_rate=0.01
counterfactual_lr_mult=0.1
counterfactual_hidden_dim=512
counterfactual_identity_dim=32
counterfactual_challenge_radius=0.5
counterfactual_adversary_strength=0.25
counterfactual_adversary_warmup_steps=500
counterfactual_robust_weight=0.25
counterfactual_alignment_weight=0.1
counterfactual_plausibility_weight=0.1
counterfactual_gain_weight=0.1
counterfactual_gain_margin=0.05
```

这意味着 DCP prompt 和分类头仍使用 `0.01`，代理模块使用
`0.01 * 0.1 = 0.001`。

## 二、反事实模块参数

### `counterfactual_enabled`

- 默认：`False`
- `counterfactual_game` 命名配置会将其设为 `True`。
- 关闭时保持原始 DCP 的缺失分支置零行为。

### `counterfactual_hidden_dim`

- 默认：`512`
- 控制方向消息、门控和挑战残差 MLP 的宽度。
- 推荐搜索：`256, 512, 1024`。
- 过小可能欠拟合图像到文本、文本到图像的条件映射；过大容易让挑战者学习潜空间攻击，并显著增加参数量。

### `counterfactual_identity_dim`

- 默认：`32`
- 编码传播方向和缺失状态。
- 推荐搜索：`16, 32, 64`。
- 两模态场景下不建议超过 `64`，身份信息本身很少，更大的维度通常只是增加冗余。

### `counterfactual_challenge_radius`

- 默认及建议首次实验：`0.5`。
- 限制挑战残差的最大 L2 范数：

  \[
  \lVert\Delta z\rVert_2\le r.
  \]

- 推荐搜索：`0.25, 0.5, 1.0, 2.0`。
- 太小：challenge 与 support 几乎相同，鲁棒损失没有意义。
- 太大：代理会离开真实模态流形，退化成普通潜空间攻击。
- 应结合 CLIP 特征范数统计解释，不能把 `1.0` 当成跨数据集通用值。

### `counterfactual_adversary_strength`

- 默认及建议首次实验：`0.25`。
- 控制梯度反转倍率，只影响挑战残差生成器接收到的分类梯度方向与大小。
- 推荐搜索：`0.1, 0.25, 0.5, 1.0`。
- 太大容易造成生成器与分类器震荡；太小则挑战者退化。
- 它与 `challenge_radius`、`robust_weight` 都会增强对抗作用，不应同时大幅提高。

### `counterfactual_adversary_warmup_steps`

- 默认：`500`。
- 前若干优化步骤把梯度反转强度从 `0` 线性增加到目标值。
- 推荐范围：总训练步数的 `5%–15%`。
- 作用是先让 support proxy 和分类器形成基本语义，再启动挑战者。

### `counterfactual_robust_weight`

- 默认及建议首次实验：`0.25`。
- 挑战代理分类损失在总损失中的权重。
- 太高时分类器可能通过忽略代理模态降低最坏情况损失。
- 需要同时观察完整模态 F1、缺失模态 F1 和模态增益损失，不能只看缺失测试集。

### `counterfactual_alignment_weight`

- 默认：`0.1`。
- 在完整图文样本上，使 `image→text` 代理靠近真实文本特征，使 `text→image` 代理靠近真实图像特征。
- 当前使用余弦对齐，它只提供配对监督，不保证代理对应自然语言或自然图像。
- 推荐搜索：`0.05, 0.1, 0.25, 0.5`。

### `counterfactual_plausibility_weight`

- 默认：`0.1`。
- 当前实现约束 challenge 靠近 support。
- 更准确的名称应理解为“局部性约束”，不是完整的语义合理性约束。
- 太低会产生潜空间攻击；太高会使 challenge 与 support 重合。
- 推荐搜索：`0.05, 0.1, 0.25, 0.5`。

### `counterfactual_gain_weight`

- 默认：`0.1`。
- 完整样本上要求真实双模态预测优于最佳单模态预测：

  \[
  L_{gain}=\max(0,L_{complete}-\operatorname{stopgrad}(L_{best\_uni})+\delta).
  \]

- `stopgrad` 防止模型通过故意恶化单模态分支满足约束。
- 如果完整模态性能下降、图文利用率降低，优先提高该权重。
- 推荐搜索：`0.05, 0.1, 0.25, 0.5`。

### `counterfactual_gain_margin`

- 默认：`0.05`。
- 要求完整双模态损失相对最佳单模态损失至少降低多少。
- 推荐搜索：`0.0, 0.02, 0.05, 0.1`。
- margin 太大时，模型可能为了制造差距而过拟合完整样本。

### `counterfactual_lr_mult`

- 默认：`0.1`。
- 代理模块实际学习率为：

  \[
  \eta_{proxy}=\eta_{base}\times\text{counterfactual\_lr\_mult}.
  \]

- 原 DCP 使用 `learning_rate=0.01`，对新增的 4M 级 MLP 通常偏大，因此默认将代理学习率降到 `0.001`。
- 推荐搜索：`0.05, 0.1, 0.2`。

## 三、原 DCP 参数

### `prompt_length`

- 默认：`36`。
- 代码中按三部分划分：相关 prompt、共享 prompt、动态 prompt，每部分约 `12` 个 token。
- 必须能够合理地被 `3` 划分，否则当前实现会因整数除法丢失长度。

### `prompt_depth`

- 默认：`6`。
- 表示注入 prompt 的 Transformer 深度。
- 推荐消融：`1, 3, 6, 9, 12`。
- 深度增加会提高参数量和传播复杂度，不保证单调提升。

### `learning_rate`

- MM-IMDb 命名配置默认：`0.01`。
- 主要用于 prompt 参数；代理模块由 `counterfactual_lr_mult` 单独缩放。
- 如果同时出现 support 对齐震荡和分类损失震荡，先降低代理倍率，不要立刻降低全部参数的学习率。

### `weight_decay`

- MM-IMDb 默认：`0.02`。
- LayerNorm 和 bias 不使用 weight decay。
- 推荐搜索：`0.01, 0.02, 0.05`。

### `warmup_steps`

- MM-IMDb 默认：`0.1`，浮点数表示总训练步数的 10%。
- 这是优化器学习率 warm-up，与对抗强度 warm-up 不同。

### `missing_ratio` 与 `missing_type`

- `missing_ratio` 分别控制 train/val/test 缺失比例。
- `missing_type` 可设置 `text`、`image`、`both`。
- 第一轮应固定训练缺失率，再跨 `0.1, 0.3, 0.5, 0.7` 测试，不要只报告单一 70% 缺失率。

## 四、推荐调参顺序

1. 关闭挑战者：`robust_weight=0`，只训练 support、alignment 和 gain，确认代理补偿本身有效。
2. 固定 `radius=0.5`，搜索 `adversary_strength` 和 `robust_weight`。
3. 检查完整模态性能是否下降；下降则提高 `gain_weight` 或降低对抗强度。
4. 再搜索 radius 与 locality/plausibility 权重。
5. 最后调整 hidden dimension、prompt depth 等容量参数。

不要一开始联合搜索全部参数。否则即使性能提升，也无法判断来自代理补偿、对抗训练还是 DCP prompt 容量变化。

## 五、必须同步监控的指标

- 完整模态 Micro/Macro-F1；
- 缺文本、缺图像、随机缺失 F1；
- support 与 challenge 的 logits 差异；
- challenge 残差实际 L2 范数；
- `proxy_gate_mean` 及分方向 gate；
- 完整模态相对最佳单模态的实际损失增益；
- support 到真实配对模态的余弦距离。

只看缺失条件下 F1 可能掩盖“模型已经退化成单模态分类器”的问题。

## 六、当前方法允许和不允许的论文表述

可以表述为：

> 学习一个摊销的局部条件挑战器，在有界代理邻域内近似搜索会改变任务决策的缺失模态表示。

不能表述为：

- 搜索了所有合理缺失模态；
- 得到了严格的类别概率上下界；
- 提供可认证鲁棒性；
- challenge 一定对应自然文本或自然图像。

当前 `support + challenge` 只得到两个前向点。要估计类别级边界，需要增加按类别或 Top-K 类别的 latent refinement；要讨论语义合理性，需要条件能量、真实模态流形约束或可解码自然性约束。
