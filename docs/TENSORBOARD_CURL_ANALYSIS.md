# TensorBoard 指标分析（curl 拉取）

> 数据来源：`curl http://localhost:6007/data/plugin/scalars/scalars?run=.&tag=<tag>`，run 为 `.`，tag 为 URL 编码的指标名（如 `loss%2Fdqn_loss`）。

---

## 一、本次拉取结果（step 104k 附近）

| 指标 | 最新 step | 最新值 | 上一档 | 说明 |
|------|-----------|--------|--------|------|
| **loss/dqn_loss** | 104000 | **0.565** | 0.558 (103k) | 近期略升，97k 处有一次明显跳升（约 0.41→0.56），与换对手后数据分布变化一致 |
| **loss/next_rank_loss** | 104000 | 1.206 | 1.205 | 稳定 |
| **loss/ding_que_ce_loss** | 104000 | 0.059 | 0.058 | 低位稳定 |
| **ding_que/match_rate** | 104000 | **71.7%** | 72.0% | 正常 |
| **hparam/lr** | 104000 | 1e-4 | 1e-4 | 未到 decay 段 |
| **test_play/avg_pt** | 100000 | **3.48** | 2.81 (95k) | 换对手后第一次 test：>3.0，优于随机 |
| **test_play/avg_ranking** | 100000 | **2.26** | 2.60 (95k) | <2.5，优于随机 |

---

## 二、结论摘要

1. **换对手效应**  
   - 约 step 97k 起 dqn_loss 从 ~0.41 升到 ~0.56，与“刚换对手”后数据分布变化一致，属正常。  
   - test_play 在 100k 第一次评估新对手：**avg_pt 3.48 > 3.0**，**avg_ranking 2.26 < 2.5**，当前模型在新对手上**优于随机**。

2. **Loss**  
   - dqn_loss 在 97k 跳升后处于 ~0.55–0.56 平台，需再观察几步是否继续上升；若持续升可考虑提前或加强 lr decay。  
   - next_rank_loss、ding_que_ce_loss 正常。

3. **定缺**  
   - match_rate ~71.7%，无需调整。

4. **建议**  
   - 再跑 1–2 个 test_every（到 105k、110k），看 test_play/avg_pt、avg_ranking 是否稳定或继续改善。  
   - 若 dqn_loss 持续上升到 >0.6，可检查 lr schedule（如 final、warm_up）或数据/对手是否过强。

---

## 三、用 curl 自行拉取命令

```bash
# 单指标（value 为 [wall_time, step, value] 的数组）
curl -s "http://localhost:6007/data/plugin/scalars/scalars?run=.&tag=loss%2Fdqn_loss"

# 主要指标
for tag in "loss/dqn_loss" "loss/next_rank_loss" "ding_que/match_rate" "test_play/avg_pt" "test_play/avg_ranking"; do
  enc=$(echo -n "$tag" | sed 's|/|%2F|g')
  echo "=== $tag ==="
  curl -s "http://localhost:6007/data/plugin/scalars/scalars?run=.&tag=${enc}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('latest:', d[-1] if d else 'no data')"
done
```

**run 名**：当前主 run 为 `.`；`/data/runs` 会列出所有 run（含部分按 tag 展开的子 run）。
