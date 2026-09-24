# RACaP Phase 2 与 Phase 1 的代码区别

## 比较口径

Phase 1 是能力课程学习结束后的冻结系统。它包括六个成熟 Policy API、Full ReAct、
Transport ReAct 和物体外观记忆，在保留的 LIBERO-90 全量评测中达到 48/90（53.3%）。
Phase 2 从这套可运行基础出发自主提出修改，并通过同 task、同 seed、同预算的成对 native
评测决定是否保留候选。Phase 2 的全量结果为 49/90（54.4%）。

## 真正进入 runtime 主链路的变化

1. **浅容器默认放置。** 当目标语义是 tray、shallow 或 pan，而且 ReAct 仍使用无 nudge、
   非居中、rim/top 这类默认参数时，wrapper 将其转换成居中的内部释放，并把 margin 至少
   设为 4.5 cm。只要 ReAct 给出显式策略，wrapper 就不覆盖。
2. **物理策略经验。** `memory/strategy_priors.json` 保存 frying pan、moka pot、mug、
   hollow vessel 和 flat package 等类别的可见条件、建议动作和不确定性规则。经验会按当前
   source 或完整 instruction 检索，而不是按 task ID 分支。
3. **Full ReAct 路由记忆。** 长指令可以临时读取 routing prior。episode 结束后 prompt 会
   恢复，避免污染无关任务。
4. **anytime 继续执行。** 对互相独立的多物体目标，一个子目标失败后可重新规划并继续，
   不再直接终止整条任务。

## 没有改变的部分

Phase 2 仍复用 `racap/policy_api/` 中的 RGB-D grounding、负反馈重新 grounding、抓取候选、
carry offset、footprint-aware insert、push、轨迹执行和 go-home。它没有读取 native predicate、
仿真 object ID 或隐藏状态。

## 结果边界

单次 LIBERO-90 全量评测的净提升只有一条，因此不能据此声称 Phase 2 在每个能力族上都
严格优于 Phase 1。更强的信号来自迁移测试：LIBERO-PRO 从 32.8% 提升到 45.0%，
LIBERO-Long 从 32.0% 提升到 46.0%，PRO median policy time 从 454 秒降到 380 秒。
这些结果说明自主进化得到的薄层主要改善了鲁棒性、工具使用和跨场景迁移，而不是简单地
记住更多 LIBERO-90 样本。
