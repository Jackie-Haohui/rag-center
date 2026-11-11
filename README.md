# RAG Center

一个轻量级的 RAG（Retrieval-Augmented Generation）能力中心，提供知识库管理、文档索引与多模式检索能力。后端基于 FastAPI，向量存储使用 PostgreSQL + pgvector，关键词检索使用 Elasticsearch，Embedding 与 LLM 均通过 OpenAI 兼容 API 接入。

## 功能特性

- **知识库管理**：按租户（`tenant_id`）创建与隔离知识库
- **文档索引**：文本分块 → Embedding 向量化 → 写入 pgvector，并异步索引到 Elasticsearch（Celery + Redis）
- **多模式检索**
  - `vector`：向量相似度检索（默认）
  - `bm25`：Elasticsearch 关键词检索
  - `hybrid`：向量 + BM25 双路召回，RRF 融合排序
- **可选专用精排**：使用 Qwen3-Rerank 对召回结果进行二次排序
- **检索日志**：记录每次检索的 query、结果与耗时
- **Web 前端**：知识库创建与批量文档上传界面

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.11+, FastAPI, SQLAlchemy, Alembic, Celery |
| 任务队列 | Redis 7 |
| 向量库 | PostgreSQL 16 + pgvector |
| 关键词检索 | Elasticsearch 8.x |
| Embedding | OpenAI 兼容 HTTP API（如 DeepSeek、OpenAI、Ollama 等） |
| LLM | OpenAI 兼容 HTTP API（用于重排，可选） |
| 前端 | React 18, Vite, Tailwind CSS, TanStack Query |

## 项目结构

```
rag-center/
├── app/                    # 后端应用
│   ├── api/v1/routes/      # API 路由
│   ├── core/               # 配置、日志、中间件、异常
│   ├── models/             # SQLAlchemy 模型
│   ├── providers/          # Embedding / LLM / 向量库 / ES / Rerank 适配层
│   ├── services/           # 业务逻辑
│   └── utils/              # 工具函数
├── frontend/               # React 前端
├── migrations/             # Alembic 数据库迁移
├── curltest/               # API 测试脚本
├── docker-compose.yml      # PostgreSQL + Elasticsearch + Redis
├── Dockerfile              # 后端容器镜像
├── .env.example            # 环境变量示例
└── pyproject.toml
```

## 快速开始

### 1. 环境要求

- Python 3.11+
- Node.js 18+（前端开发）
- Docker & Docker Compose

### 2. 启动基础设施

```bash
docker compose up -d
```

将启动：

- **PostgreSQL + pgvector**：`localhost:5432`，数据库 `rag_center`
- **Elasticsearch**：`localhost:9200`
- **Redis**：`localhost:6379`（Celery Broker）

### 3. 配置环境变量

```bash
cp .env.example .env
```

按需编辑 `.env`，关键配置示例：

```env
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_API_KEY=your-api-key
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIMENSIONS=1024

LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=deepseek-v4-flash
```

> **注意**：`EMBEDDING_DIMENSIONS` 必须与 Embedding 模型输出维度一致。更换模型或维度后，需清空并重建已有索引数据。

### 4. 安装 Python 依赖（首次初始化）

项目依赖定义在 `pyproject.toml` 中，在项目根目录执行：

```bash
pip install -e .
```

会以「可编辑模式」安装本项目及其依赖（FastAPI、SQLAlchemy 等），之后改代码无需重复安装。

### 5. 初始化数据库表结构（首次初始化）

PostgreSQL 容器启动后数据库是空的，需要执行一次 Alembic 迁移来建表：

```bash
alembic upgrade head
```

这一步只需在**第一次启动**或**项目数据库结构有升级**时执行，不是每次启动都要跑。

### 6. 启动后端与 Celery Worker

文档索引在后台异步执行，需要**同时启动 API 和 Worker**。

终端 1 — API：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

终端 2 — Celery Worker：

```bash
celery -A app.celery_app worker --loglevel=info
```

启动后访问 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 查看 Swagger 文档。

> 未启动 Worker 时，upload 仍会返回 `PROCESSING`，但索引不会执行。

### 7. 启动前端（可选）

```bash
cd frontend
cp .env.example .env   # 填入 API_KEY（见下方说明）
npm install
npm run dev
```

前端默认运行在 [http://127.0.0.1:5173](http://127.0.0.1:5173)，API 请求通过 Vite 代理转发到后端 `8000` 端口。

开发前在 `frontend/.env` 中配置 `API_KEY`（由 `scripts/create_api_key.py` 生成；后端 `AUTH_ENABLED=true` 时必填）。本地免 Key 调试可设后端 `AUTH_ENABLED=false`，前端 `API_KEY` 留空即可。

## 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | PostgreSQL 连接串 | `postgresql+psycopg://postgres:postgres@localhost:5432/rag_center` |
| `MODEL_BASE_URL` | Embedding API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `MODEL_API_KEY` | Embedding API Key | — |
| `EMBEDDING_MODEL` | Embedding 模型名 | `text-embedding-v3` |
| `EMBEDDING_DIMENSIONS` | 向量维度 | `1024` |
| `EMBEDDING_BATCH_SIZE` | Embedding 批处理大小 | `10` |
| `LLM_BASE_URL` | LLM API 地址 | `https://api.deepseek.com/v1` |
| `LLM_API_KEY` | LLM API Key | — |
| `LLM_MODEL` | 快速 LLM 模型名 | `deepseek-v4-flash` |
| `LLM_STRONG_MODEL` | 强 LLM 模型名 | `deepseek-v4-pro` |
| `VECTOR_STORE` | 向量存储后端 | `pgvector` |
| `KEYWORD_SEARCH_PROVIDER` | 关键词检索后端 | `elasticsearch` |
| `ELASTICSEARCH_URL` | Elasticsearch 地址 | `http://localhost:9200` |
| `EXTERNAL_HTTP_MAX_CONNECTIONS` | 单进程外部 HTTP 最大连接数 | `100` |
| `EXTERNAL_HTTP_MAX_KEEPALIVE_CONNECTIONS` | 单进程外部 HTTP 保活连接数 | `20` |
| `EXTERNAL_HTTP_KEEPALIVE_EXPIRY_SECONDS` | 外部 HTTP 保活连接有效期（秒） | `30` |
| `RETRIEVAL_MODE` | 默认检索模式 | `vector` |
| `TOP_K` | 默认返回条数 | `5` |
| `CHUNK_SIZE` | 分块大小（字符） | `800` |
| `CHUNK_OVERLAP` | 分块重叠（字符） | `100` |
| `RERANK_ENABLED` | 是否启用专用精排 | `false` |
| `RERANK_BASE_URL` | 百炼 Qwen3-Rerank Workspace 接口地址 | — |
| `RERANK_API_KEY` | 可选精排 Key，留空复用 `MODEL_API_KEY` | — |
| `RERANK_MODEL` | 精排模型名称 | `qwen3-rerank` |
| `RERANK_TIMEOUT_SECONDS` | 精排接口超时秒数 | `15` |
| `QUERY_REWRITE_ENABLED` | 是否默认启用 query 改写 | `false` |
| `QUERY_REWRITE_TIMEOUT_MS` | query 改写 LLM 超时（毫秒） | `2000` |
| `CELERY_BROKER_URL` | Celery Broker（Redis） | `redis://localhost:6379/0` |
| `CELERY_RESULT_BACKEND` | Celery 结果后端 | `redis://localhost:6379/1` |

完整配置见 [`.env.example`](.env.example)。

## API 接口

所有接口前缀为 `/api/v1`，响应格式统一为：

```json
{
  "code": 0,
  "msg": "success",
  "data": { ... }
}
```

业务接口需在请求头携带 API Key（本地调试可设 `AUTH_ENABLED=false` 跳过）：

```bash
-H "Authorization: Bearer rk_live_你的key"
```

可先调用 `GET /api/v1/auth/me` 验证 Key 是否有效。租户身份由 Key 决定，请求体**无需**再传 `tenant_id`。

### 创建知识库

`POST /api/v1/knowledge-bases/create`

```bash
curl -X POST http://127.0.0.1:8000/api/v1/knowledge-bases/create \
  -H "Authorization: Bearer rk_live_你的key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "退款政策知识库",
    "description": "用于客服退款问题问答"
  }'
```

### 上传文档

`POST /api/v1/documents/upload`

接口会立即返回 `status=PROCESSING`（异步索引），后台 Celery Worker 完成分块、向量化并写入 pgvector 与 Elasticsearch。可通过下方「查询文档状态」接口轮询结果。

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -H "Authorization: Bearer rk_live_你的key" \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": "<上一步返回的 kb_id>",
    "title": "退款政策",
    "content": "用户可在订单完成后 7 天内申请退款。特殊商品不支持无理由退款。退款会在 3 个工作日内到账。"
  }'
```

### 查询文档状态

`GET /api/v1/documents/{document_id}`

`status`：`1=SUCCESS`，`2=FAILED`，`3=PROCESSING`。

```bash
curl http://127.0.0.1:8000/api/v1/documents/<document_id> \
  -H "Authorization: Bearer rk_live_你的key"
```

### 知识库运维

`GET /api/v1/knowledge-bases/{kb_id}` — 查详情（含 settings）

`PATCH /api/v1/knowledge-bases/{kb_id}` — 改名称 / 描述 / 词表

`DELETE /api/v1/knowledge-bases/{kb_id}` — 删库及全部文档

```bash
curl http://127.0.0.1:8000/api/v1/knowledge-bases/<kb_id> \
  -H "Authorization: Bearer rk_live_你的key"

curl -X PATCH http://127.0.0.1:8000/api/v1/knowledge-bases/<kb_id> \
  -H "Authorization: Bearer rk_live_你的key" \
  -H "Content-Type: application/json" \
  -d '{"name":"新名称"}'

curl -X DELETE http://127.0.0.1:8000/api/v1/knowledge-bases/<kb_id> \
  -H "Authorization: Bearer rk_live_你的key"
```

### 文档运维

`DELETE /api/v1/documents/{document_id}` — 删文档

`POST /api/v1/documents/{document_id}/reindex` — 重建索引（**SUCCESS** 或 **FAILED** 文档；`PROCESSING` 中不可操作）

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/documents/<document_id> \
  -H "Authorization: Bearer rk_live_你的key"

curl -X POST http://127.0.0.1:8000/api/v1/documents/<document_id>/reindex \
  -H "Authorization: Bearer rk_live_你的key"
```

文档改内容：先 DELETE，再重新 upload。

### 切块策略变更后 reindex

优化结构化切块（按标题/表格）上线后，**仅新 upload 的文档**会用新规则；已有库需 **全量 reindex** 才会替换旧 chunk。

```bash
# 整库重建（需 Celery Worker 运行）
python scripts/reindex_knowledge_base.py --kb-id <kb_id> --wait

# 指定单篇
python scripts/reindex_knowledge_base.py --kb-id <kb_id> --document-id <document_id> --wait
```

脚本会 purge pgvector / ES 中的旧 chunk，从 `documents.content` 重新索引，无需重新上传文件。

### 检索

`POST /api/v1/rag/retrieve`

可通过 **`profile`** 选择检索策略（与租户套餐 plan 联动，超限返回 `20013` / `20014` / `20005`）：

| profile | 说明 | 典型 plan |
|---------|------|-----------|
| `speed` | 向量检索，最快 | free 可用 |
| `balanced` | hybrid，均衡（**未传 profile 时默认**） | standard / pro |
| `quality` | hybrid + rerank + query 改写 | 仅 pro |
| `custom` | 自行传 `retrieval_options` / `rerank_options`，仍受 plan 约束 | standard / pro |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/retrieve \
  -H "Authorization: Bearer rk_live_你的key" \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": "<kb_id>",
    "user_id": "user_demo",
    "query": "退款需要几天内申请？",
    "profile": "balanced"
  }'
```

**多库联合检索**（`kb_ids` 数组；仍兼容只传 `kb_id`）：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/retrieve \
  -H "Authorization: Bearer rk_live_你的key" \
  -H "Content-Type: application/json" \
  -d '{
    "kb_ids": ["<kb_ecommerce>", "<kb_customer_service>"],
    "user_id": "user_demo",
    "query": "退款时限和会员补偿标准",
    "profile": "balanced",
    "top_k": 5
  }'
```

各库并行召回后经跨库 RRF 融合；每条 chunk 带 `kb_id` / `kb_name`。一次多库 retrieve 仍计 **1 次** 日检索量。套餐限制：`free` 仅单库，`standard` 最多 3 库，`pro` 最多 5 库（超限 `20013`）。

**pro 追求质量：**

```json
{
  "kb_id": "<kb_id>",
  "user_id": "user_demo",
  "query": "退款需要几天内申请？",
  "profile": "quality"
}
```

响应 `metadata.tenant_policy` 会返回本次 plan、profile 及实际生效参数摘要。

仍支持通过 `retrieval_options` 覆盖检索模式（`profile=custom` 时）：

```json
{
  "kb_id": "<kb_id>",
  "user_id": "user_demo",
  "query": "退款需要几天内申请？",
  "top_k": 5,
  "retrieval_options": {
    "mode": "hybrid",
    "vector_top_k": 20,
    "bm25_top_k": 20,
    "rrf_k": 60
  },
  "rerank_options": {
    "enabled": true,
    "top_n": 5
  }
}
```

`retrieval_options.mode` 可选值：`vector` | `bm25` | `hybrid`。

可选通过 `query_options` 开启 LLM query 改写（默认关闭）：

```json
{
  "kb_id": "<kb_id>",
  "user_id": "user_demo",
  "query": "背调要问啥",
  "query_options": {
    "enabled": true,
    "strategy": "rewrite"
  }
}
```

响应 `metadata.query_processing` 会返回原话、实际检索句及改写耗时；改写失败时自动降级为原话检索。

### 套餐相关错误码

| code | 含义 | 典型场景 |
|------|------|----------|
| `20013` | 当前套餐不支持该功能 | free 传 `profile=balanced`；standard 开 rerank；free 传多个 `kb_ids` |
| `20014` | 配额已用尽 | free 建第 2 个库；日检索量超限 |
| `20005` | 接口调用超限 | 秒级 QPS 超限 |

租户 plan 由 `scripts/update_tenant_plan.py` 调整，详见 `V2/优化九：多租户差异化服务/`。

### 知识库词表（query 扩展）

每个知识库可单独配置 `settings.synonyms`，在检索前自动把业务术语补进 query（**不**走文档 upload，**不**消耗 LLM）。词表按 kb 维护，默认无配置时不生效。

示例配置见 [`examples/kb_settings.example.json`](examples/kb_settings.example.json)。写入方式：

```bash
python scripts/update_kb_settings.py \
  --kb-id <kb_id> \
  --settings-file examples/kb_settings.example.json
```

脚本会 **整文件覆盖** 该 kb 的 `settings` 字段。检索响应 `metadata.query_processing` 在词表命中时会返回 `search_query`、`synonym_expansions`；未开 LLM 改写时词表仍会生效。

更多示例脚本见 [`curltest/`](curltest/) 目录。

租户接入、plan 与 profile 说明见 [`docs/tenant-onboarding.md`](docs/tenant-onboarding.md)。

## 离线测评（RAGAS）

改词表、改 profile 前，用固定 golden set 批量跑 retrieve，RAGAS 输出 `context_precision` / `context_recall`，改配置前后对比报告。

### 1. 安装评测依赖

```bash
pip install -e ".[eval]"
```

需在 `.env` 配置 `LLM_API_KEY`（或 `MODEL_API_KEY`）供 RAGAS 评判；retrieve 仍走 RAG Center API Key。

### 2. 从 Langfuse 导出低分反馈（主数据源）

```bash
python scripts/export_eval_cases_from_langfuse.py \
  --max-score 3 \
  --days 30 \
  --kb-id <kb_id> \
  --output eval/datasets/imported_from_feedback.json
```

导出 case 的 `ground_truth` 为空，需打开知识库文档摘关键句。

**单独跑反馈集**（补完 ground_truth 后建议先做，不必 merge）：

```bash
python scripts/run_retrieval_eval.py \
  --dataset eval/datasets/imported_from_feedback.json \
  --api-key rk_live_你的key \
  --profile balanced \
  --output eval/reports/imported_balanced.json
```

可选 merge 进主集：

```bash
python scripts/export_eval_cases_from_langfuse.py \
  --max-score 3 \
  --merge eval/datasets/ecommerce_retrieval.json \
  --output eval/datasets/ecommerce_retrieval.json
```

冷启动可直接用 [`eval/datasets/ecommerce_retrieval.json`](eval/datasets/ecommerce_retrieval.json)（将 `kb_id` 换成实际值）。

### 3. 全量 RAGAS 评测（主集回归）

```bash
python scripts/run_retrieval_eval.py \
  --dataset eval/datasets/ecommerce_retrieval.json \
  --api-key rk_live_你的key \
  --profile balanced \
  --output eval/reports/balanced.json
```

改词表 / profile 后重跑**同一 `--dataset`**，对比 `eval/reports/` 下前后分数。

详见 [`eval/datasets/`](eval/datasets/) 与 `V2/优化十：检索效果调优/03-离线测评.md`。

## Docker 部署后端

```bash
docker build -t rag-center .
docker run --env-file .env -p 8000:8000 rag-center
```

需确保 PostgreSQL 与 Elasticsearch 可从容器内访问（修改 `.env` 中的连接地址或使用同一 Docker 网络）。

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码检查
ruff check .

# 新建数据库迁移
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```
