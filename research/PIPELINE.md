# Chekhov 自动研究流程(PIPELINE.md)

本文档是给 Claude Code 的执行指令,参考 SakanaAI/AI-Scientist-v2 的架构改造而来:
所有原本通过 API 调用 LLM 完成的角色(ideation、实验编排、写代码、调试、画图、写作、审稿),
均由 Claude Code 本人及其子 agent 承担,不使用任何外部 LLM API。
文献检索用 WebSearch/WebFetch 替代 Semantic Scholar API。

启动方式(用户输入示例):

> 读 research/PIPELINE.md,基于 research/ideas/<topic>.md 开始一轮研究。

一旦确认启动,进入自主模式:**不要中途停下来询问是否继续**。用户可能在睡觉,
流程持续运行直到全部阶段完成或被手动打断。

## 配置(默认值,可在启动时由用户覆盖)

| 参数 | 默认 | 含义 |
|---|---|---|
| NUM_IDEAS | 5 | ideation 阶段生成的候选想法数 |
| NUM_REFLECTIONS | 3 | 每个想法的自我批判/改进轮数 |
| NUM_DRAFTS | 3 | Stage 1 并行独立初稿实现数(对应原版 num_drafts) |
| MAX_DEBUG_DEPTH | 3 | 一个失败节点最多连续调试次数,超过则放弃该路径 |
| TUNING_ITERS | 8 | Stage 2 调优迭代次数 |
| NUM_ABLATIONS | 4 | Stage 3 消融实验数 |
| RUN_TIMEOUT_MIN | 15 | 单次实验运行超时(分钟),超时即杀掉并记为失败 |
| GPU_POLICY | 空闲卡 | 用 nvidia-smi 选空闲 GPU,多节点并行时各占一张卡 |

## 一轮研究(run)的目录结构

```
research/runs/<YYYYMMDD>_<idea-slug>/
  idea.md            — 选定的研究想法(假设、实验方案、评价指标、预期结果)
  journal.jsonl      — 实验日志,每个节点一行(见"节点协议")
  nodes/<node_id>/   — 每个实验节点:experiment.py、run.log、metrics.json、notes.md
  best/BEST.md       — 指向当前最优节点的记录(node_id + 指标 + 原因)
  figures/           — 汇总图表(matplotlib,保存 png)
  paper/paper.md     — 论文(Markdown;若环境有 pandoc/pdflatex 则另编译 PDF)
  review.md          — 自我审稿意见与修订记录
```

## 阶段 0:环境自检与立项

1. 检查 GPU(`nvidia-smi`)、磁盘空间、Python 版本。
2. 在 run 目录下创建 venv 并安装该实验所需依赖(torch 等),缓存复用已有环境。
   依赖装不上就换更小的依赖方案,不要卡死。
3. 读取指定的 topic 文件,创建 run 目录,初始化 journal.jsonl。

## 阶段 1:Ideation(想法生成)

对应原版 perform_ideation_temp_free.py:

1. 读 topic 文件(Title / Keywords / TL;DR / Abstract)。
2. 生成 NUM_IDEAS 个候选想法,每个包含:假设、为什么可能成立、最小可行实验设计、
   评价指标(必须是单一可比较的标量)、风险。
3. 每个想法做 NUM_REFLECTIONS 轮自我批判:可行性(单机 GPU、时限内)、新颖性、清晰度。
4. 新颖性检查:用 WebSearch 搜索相关工作;明显已被做烂的想法降级或淘汰。
5. 按「可行性 × 有趣程度」排序,选出第 1 名写入 idea.md,其余留档 ideas_candidates.md。

## 阶段 2:实验树搜索(简化版 BFTS)

对应原版 launch_scientist_bfts.py + treesearch,分三个子阶段。
每个"节点" = 一份完整可运行的 experiment.py + 它的运行结果。

**Stage 2.1 初稿(广度)**:用 Agent 工具并行派 NUM_DRAFTS 个子 agent,各自独立实现
idea.md 的最小实验(互相不通气,鼓励实现路线差异)。每个子 agent 在自己的 nodes/<id>/
目录下工作,跑通并输出 metrics.json。失败节点按 MAX_DEBUG_DEPTH 调试,修不好就放弃。
全部结束后,按指标选出最优节点作为主干。

**Stage 2.2 调优(深度)**:在最优节点上迭代 TUNING_ITERS 次:
每次提出一个改动(超参、方法变体、训练技巧),新建子节点目录,跑,比指标。
更好 → 成为新主干;更差 → 留档,回到原主干。连续 3 次无改进时,允许尝试一次
更激进的改动(相当于原版的探索分支)。

**Stage 2.3 消融(严谨性)**:对最终主干做 NUM_ABLATIONS 个消融:逐一去掉/替换
关键组件,验证每个组件确实有贡献。消融结果不改变主干,只记录进 journal。

### 节点协议

- 运行方式:`cd nodes/<id> && <venv-python> experiment.py > run.log 2>&1`,
  绝不让训练输出直接进入上下文;只 grep/读取关键行。
- experiment.py 必须在结束时把结果写成 metrics.json(含主指标)并打印一行
  `FINAL_METRIC: <value>`。
- 每个节点跑完后向 journal.jsonl 追加一行:
  `{"node": id, "parent": id|null, "stage": "draft|tune|ablation", "status": "ok|fail|timeout", "metric": float|null, "change": "一句话描述", "ts": "ISO时间"}`
- 超时 RUN_TIMEOUT_MIN 分钟即 kill,status=timeout。
- 崩溃时读 run.log 末尾 50 行定位;小问题(typo、shape 不匹配)就地修,
  想法本身不成立就放弃节点。

## 阶段 3:画图汇总

对应原版 perform_plotting.py:从 journal.jsonl 和各节点 metrics.json 汇总,
用 matplotlib 生成论文用图(主结果对比、调优轨迹、消融柱状图),存 figures/。
图要有轴标签、图例、标题,风格统一。

## 阶段 4:写论文

对应原版 perform_writeup.py:写 paper/paper.md,结构:
Abstract / Introduction / Related Work(用 WebSearch 找 5–10 篇真实文献,给出真实
链接,严禁编造引用)/ Method / Experiments(引用 figures/ 的图)/ Ablations /
Limitations / Conclusion。
诚实报告:负结果照实写,不夸大。若有 pandoc 或 pdflatex,编译一份 PDF。
论文中必须声明由 AI 自动生成。

## 阶段 5:自我审稿与修订

对应原版 perform_llm_review.py:以顶会审稿人身份写 review.md,按
「清晰度 / 新颖性 / 实验严谨性 / 结论是否被证据支撑」打分并列出具体缺陷。
然后做一轮修订:能用现有数据修的直接改论文;需要补实验的,若总时长允许则补跑
(走节点协议),否则如实写进 Limitations。

## 阶段 6:收尾报告

给用户一份简短总结:研究问题、最终结论(含主指标数字)、最优节点相对基线的提升、
论文与图表路径、以及"下一步值得做什么"。

## 硬性规则

1. 只在 run 目录内写实验文件;不修改 research/ 之外的项目代码。
2. 一切结果以落盘文件为准(journal.jsonl 是唯一事实来源),不依赖上下文记忆。
3. 长输出一律重定向到文件,上下文里只保留摘要。
4. 实验代码出自己手也要怀疑:指标异常好时先查数据泄漏/评测 bug,再庆祝。
5. 不编造:引用、数字、图表全部可溯源到文件或真实网页。
6. 自主模式下不停顿询问;唯一例外是磁盘/GPU 出现可能影响机器上其他用户的异常。
