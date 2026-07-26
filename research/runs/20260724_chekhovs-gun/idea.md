# Idea: Chekhov's Gun in LLMs — Priming-Induced Lexical Intrusion

## 核心假设 (H1)
当上下文早期出现某个主题/实体(“契诃夫之枪”),即使它与当前实际询问的问题
无关,LLM 在生成回答时也倾向于把这个先前提及的关键词/主题带出来使用
(lexical/topical intrusion)。类比戏剧中“第一幕挂在墙上的枪必然会开火”。

## 判别式对照设计
对每一条测试项构造一对最小差异输入(paired, minimal-difference):
- **Treatment(有枪)**:上下文中被植入一个无关的显著实体/主题 D(distractor)。
- **Control(无枪)**:同样的上下文,但 D 被替换为中性内容或移除,其余完全相同。
两者被问的**目标问题 Q 完全相同**,且 Q 的正确回答**与 D 无关**。
若模型存在契诃夫之枪效应,则 Treatment 组回答中出现 D(或其近义/词形变体)的
比例应显著高于 Control 组。

## 两个任务范式
1. **多轮对话 (multi-turn dialogue)**:早期轮次用户顺带提到无关主题 D
   (如“对了我最近在养多肉植物”),随后若干轮讨论别的事,最后问一个与 D
   无关的开放问题 Q。测 D 是否被无端带入最终回答。变量:D 与 Q 之间的轮次距离。
2. **阅读理解 (reading comprehension)**:给一段文段,文段中包含一个显著但与
   问题无关的干扰实体 D(Treatment)vs 不含 D(Control),问一个答案不涉及 D
   的问题 Q。测 D 是否被无端写入回答。

## 指标
- **主指标 intrusion_rate**:回答中出现植入关键词(词形归一化后匹配)的比例。
  效应量 = rate(Treatment) − rate(Control)。
- **配对显著性**:McNemar 检验 + bootstrap 95% CI(按 item 配对)。
- **机制指标(H2)**:在回答生成位置上,植入词首 token 的 log-prob 在
  Treatment 相对 Control 的提升 Δlogp;及其随对话/文本距离的衰减曲线。

## 受试模型
- 主模型:Qwen3-8B(本地,关闭 thinking 模式)。
- 若时间允许:Qwen3.5-4B / Llama-3.1-8B-Instruct 做跨模型验证(消融)。

## 消融 (H3)
- 距离:D 距 Q 越远,intrusion 是否衰减?
- 显著性:D 的 surprise/罕见度是否放大 intrusion?
- 指令强度:system prompt 明确要求“只回答被问的问题”能否抑制 intrusion?
- 模型规模:跨模型规模是否单调?

## 约束
- 仅 GPU 6,7;单实验 ≤15 min;transformers 本地推理;不用任何外部 API。
- 主结果需可配对、可统计检验,负结果如实报告。
- 论文:ACL 2023 模板,长文 8 页,声明由 AI 自主生成。
