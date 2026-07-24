# Title

Data Ordering and Curriculum Effects in Small-Scale Language Model Training

# Keywords

curriculum learning, data ordering, small language models, sample efficiency, training dynamics

# TL;DR

在固定 token 预算下,训练数据的呈现顺序(随机 vs. 由易到难 vs. 其他排序策略)
是否影响小型 Transformer 语言模型的最终验证损失?

# Abstract

大规模语言模型训练普遍采用随机打乱的数据顺序,但在小模型、小预算的场景下,
数据课程(curriculum)的效果仍有争议。本主题旨在系统研究:在单卡 GPU、
固定训练 token 预算(如 10 分钟内可完成)的约束下,不同的数据排序策略
(随机、按序列难度升序/降序、按困惑度分桶等)对小型 Transformer
(1000 万至 5000 万参数)最终验证损失的影响。实验应使用公开的小型文本数据集
(如 TinyStories 或 enwik8 的子集),以验证集 loss(或 bits per byte)为
唯一主指标,并通过消融确认效果来源。任何方向的明确结论(包括"没有显著差异"
的负结果)都有价值。

# Constraints

- 单卡 A100 80GB,单次实验含训练评估在 15 分钟内完成。
- 只用 PyTorch 与常见 Python 库,不依赖外部 LLM API。
- 主指标:固定 token 预算下的验证集 loss(越低越好)。
