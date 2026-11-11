# 09 - Research 深度研究

> **Research (深度研究)**  
> Research 是 rag-center 最强的检索能力。它不是一次简单的检索，而是一个多阶段的自动化分析流程：规划 → 检索 → 决策 → 补充 → 生成。

---

## 1. Research 是什么？

### 1.1 与普通 Retrieve 的区别

| 维度 | Retrieve | Research |
|------|---------|----------|
| 轮次 | 1次 | 最多2轮 (可扩展) |
| LLM参与 | 可选 (query rewrite) | 必须 (3个LLM角色) |
| 输出 | chunks + 元数据 | answer + evidence_pack |
| 问题拆解 | 无 | Planner自动拆解 |
| 证据评估 | 无 | Decider判断完整性 |
| 补充检索 | 无 | Supplement自动补搜 |
| 可用套餐 | 全部 | Pro only |

### 1.2 适用场景

```
✅ 复杂多维度问题:
   "P7级别员工在哺乳期的考勤规定和薪酬保护政策是什么？"

✅ 对比分析:
   "标准档和专业档在检索功能上的完整差异对比"

✅ 需要证据链:
   "根据公司规定，请给出因公出差的报销完整流程和所需材料"

❌ 简单事实查询:
   "公司地址在哪里？" → 用普通 Retrieve 即可
```

---

## 2. Research 五阶段流程

```
┌────────────────────────────────────────────────────────┐
│              Research Pipeline (Pro套餐)                │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Stage 1: PLANNING     [Planner - 快速LLM]              │
│    输入: 用户问题                                        │
│    输出: problem_type + aspects + initial_queries       │
│    例: "P7产假政策"                                      │
│      → aspects: ["产假天数", "薪酬发放", "申请流程"]     │
│      → queries: ["P7产假天数规定", "产假期间薪酬...", ...]│
│                                                         │
│  Stage 2: RETRIEVING   [并行检索]                       │
│    每个 query → 完整检索流水线 (hybrid+rerank+context)    │
│    例: 3个queries × 1次完整检索 = 3个并行任务            │
│                                                         │
│  Stage 3: DECIDING     [Decider - 快速LLM]              │
│    输入: 所有检索结果 + aspects                          │
│    输出: evidence_pack + missing_aspect_ids + next_queries│
│    判断: "产假天数" ✅ / "薪酬发放" ⚠️部分 / "申请流程" ❌ │
│                                                         │
│  Stage 4: SUPPLEMENT   [补充检索] (有缺失时触发)         │
│    对 missing aspects 生成新查询, 再次并行检索            │
│                                                         │
│  Stage 5: FINALIZING   [Finalizer - 强LLM]              │
│    输入: 完整候选集 + evidence_pack                       │
│    输出: final_answer + final_evidence_pack              │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### 2.1 代码实现

```python
# app/services/research_service.py
async def run(self, *, state, request, knowledge_bases, allow_supplement=True):
    started_at = perf_counter()

    # Stage 1: Planning
    state.stage = "planning"
    planner_result = await self.planner.plan(
        question=state.original_query,
        knowledge_bases=knowledge_bases,
    )
    # 输出: problem_type, aspects, initial_queries

    # Stage 2: Retrieving (Round 1)
    state.stage = "retrieving"
    round_one = await self.parallel_retrieval.execute(
        state=state,
        queries=planner_result.queries,
        request=request,
        knowledge_bases=knowledge_bases,
        round=1,
    )
    first_fusion = self.fusion.fuse(results=round_one.results, aspects=state.aspects)

    # Stage 3: Deciding
    state.stage = "deciding"
    decider_result = await self.decider.decide(
        state=state,
        candidates=first_fusion.candidates,
    )
    # 输出: evidence_pack, missing_aspect_ids, next_queries

    # Stage 4: Supplement (如果有缺失)
    if decider_result.pack.status != "complete" and allow_supplement:
        state.stage = "supplementing"
        supplement_result = await self.supplement.execute(
            state=state,
            queries=decider_result.next_queries,
            ...
        )
        final_fusion = self.fusion.fuse(results=all_results, ...)

    # Stage 5: Finalizing
    if 未超时 and 未超LLM预算:
        state.stage = "finalizing"
        finalizer_result = await self.finalizer.finalize(
            state=state,
            candidates=final_fusion.candidates,
            fallback_pack=decider_result.pack,
        )
        answer = finalizer_result.answer
        pack = finalizer_result.pack

    return ResearchRunResult(state=state, answer=answer, evidence_pack=pack, ...)
```

---

## 3. LLM 角色分工

Research 使用了两个级别的 LLM：

```python
class LLMRole(Enum):
    FAST = "fast"       # 便宜快速, 用于 Planner/Decider
    STRONG = "strong"   # 贵但精准, 用于 Finalizer

# Planner:   快速LLM → 理解问题, 拆解方向, 生成检索查询
# Decider:   快速LLM → 判断证据完整性, 生成补充查询
# Finalizer: 强LLM   → 基于完整证据包生成最终答案
```

---

## 4. 硬限制与预算控制

```python
# app/services/research_limits.py
DEFAULT_RESEARCH_TIMEOUT_SECONDS  # 整体超时
MAX_RESEARCH_ROUNDS = 2           # 最多2轮检索
MAX_RESEARCH_LLM_CALLS            # 最多LLM调用次数

# 触发停止的条件:
state.stop_reason 可能的值:
  "evidence_complete"  → 证据完整, 正常结束
  "max_rounds"         → 达到最大轮次
  "timeout"            → 超时
  "budget_exhausted"   → LLM调用次数耗尽
  "degraded"           → 某阶段降级
  "duplicate_query"    → 补充查询与已执行查询重复
  "no_new_evidence"    → 补充检索无法找到新证据
```

---

## 5. Research API

### 5.1 请求

```bash
POST /api/v1/rag/research
Authorization: Bearer <your_api_key>
Content-Type: application/json

{
    "kb_id": "kb-hr",
    "user_id": "user_001",
    "query": "P7级别员工在哺乳期的考勤规定和薪酬保护政策"
}
```

### 5.2 响应 (简化)

```json
{
    "code": 0,
    "data": {
        "query": "P7级别员工在哺乳期的考勤规定...",
        "answer": "根据公司规定，P7级别员工在哺乳期享有以下权益：\n\n1. 考勤方面...",
        "evidence_pack": {
            "status": "complete",
            "groups": [
                {
                    "aspect": "哺乳期考勤规定",
                    "evidence": [
                        {"role": "core", "evidence_id": "E1"},
                        {"role": "supporting", "evidence_id": "E2"}
                    ],
                    "covered": true
                }
            ],
            "items": [...]
        },
        "plan": {
            "problem_type": "policy_lookup",
            "aspects": [
                {"aspect_id": "A1", "description": "哺乳期考勤规定"},
                {"aspect_id": "A2", "description": "哺乳期薪酬保护"}
            ],
            "initial_queries": [
                "P7级别 哺乳期 考勤规定",
                "哺乳期员工 薪酬保护政策 P7"
            ]
        },
        "rounds": [
            {
                "round": 1,
                "purpose": "initial",
                "candidate_count": 35,
                "tasks": [
                    {
                        "query_id": "q1",
                        "query": "P7级别 哺乳期 考勤规定",
                        "success": true,
                        "chunk_count": 12,
                        "latency_ms": 245
                    }
                ]
            }
        ],
        "metadata": {
            "research_id": "...",
            "trace_id": "...",
            "round_count": 1,
            "retrieval_task_count": 2,
            "llm_call_count": 3,
            "latency_ms": 2850,
            "stop_reason": "evidence_complete",
            "degraded": false
        }
    }
}
```

---

## 6. 并行检索

```python
# app/services/research_parallel_retrieval_service.py
class ResearchParallelRetrievalService:
    async def execute(self, *, state, queries, request, knowledge_bases, round):
        # 多个查询并行执行完整检索流水线
        tasks = [
            self.retrieve_once.execute(
                tenant_id=..., request=request,
                knowledge_bases=knowledge_bases,
                search_query=query,
                query_id=f"r{round}_q{i}",
                aspect_ids=[...],
                round=round,
            )
            for i, query in enumerate(queries)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 如果某个task失败, 标记为degraded但不影响其他task
        return ParallelRetrievalResult(results=results, ...)
```

---

## 7. 候选融合

```python
# app/services/research_candidate_fusion_service.py
class ResearchCandidateFusionService:
    def fuse(self, *, results, aspects):
        all_candidates = []
        for result in results:
            all_candidates.extend(result.candidate_snapshot)

        # 去重: (kb_id, chunk_id) 唯一
        # 排序: 取每个chunk的最佳rank
        # 限制: 最多保留一定数量
        seen = {}
        for candidate in all_candidates:
            key = (candidate["kb_id"], candidate["chunk_id"])
            if key not in seen or candidate.get("best_rank", 99) < seen[key].get("best_rank", 99):
                seen[key] = candidate

        return CandidateFusionResult(candidates=list(seen.values()))
```

---

## 8. 可观测性

Research 有完整的 Langfuse 追踪：

```python
# app/observability/research_observability.py
class ResearchObservability:
    # 记录每个阶段:
    - start()          # 开始 trace + generation
    - record_planner() # Planner的输入输出+用量
    - record_round()   # 每轮检索的详细结果
    - record_decider() # Decider的决策
    - record_finalizer() # Finalizer的生成
    - finish()         # 结束, 记录总耗时和状态
```

---

## 下一步阅读

- [10 - 多租户权限与套餐体系](./10-多租户权限与套餐体系.md)
- [06 - 检索模式详解](./06-检索模式详解.md)