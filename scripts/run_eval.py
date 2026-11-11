#!/usr/bin/env python3
"""Unified offline evaluation entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("EVAL_LANGFUSE_ENABLED", "false")
os.environ.setdefault("LANGFUSE_ENABLED", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.core.dataset import DatasetValidationError, load_dataset
from eval.core.experiment import ExperimentValidationError, load_experiment
from eval.core.metrics import reconcile_run_identical_ragas_scores, score_run_directory
from eval.core.report import generate_report
from eval.core.runner import (
    EvaluationRunError,
    retry_failed_shared_rerank,
    run_retrieval_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rag-center 统一离线评测")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("validate-dataset", help="校验 Golden Set")
    dataset_parser.add_argument("--dataset", type=Path, required=True)

    experiment_parser = subparsers.add_parser(
        "validate-experiment",
        help="校验 baseline/candidate 实验配置",
    )
    experiment_parser.add_argument("--experiment", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="运行检索、评分并生成报告")
    run_parser.add_argument("--experiment", type=Path, required=True)
    run_parser.add_argument("--api-key", default=None)
    run_parser.add_argument(
        "--tenant-id",
        default=None,
        help="本地评测租户；未提供 API Key 时创建并自动删除临时 Key",
    )
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "eval" / "results",
    )
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--case-id", action="append", dest="case_ids", default=None)
    run_parser.add_argument("--output-suffix", default=None)
    run_parser.add_argument("--no-progress", action="store_true")

    score_parser = subparsers.add_parser("score", help="对已保存的原始结果重新评分")
    score_parser.add_argument("--run-dir", type=Path, required=True)
    score_parser.add_argument("--no-progress", action="store_true")
    score_parser.add_argument(
        "--missing-only",
        action="store_true",
        help="复用已有成功分数，只补评缺失的 RAGAS 输入",
    )

    report_parser = subparsers.add_parser("report", help="根据已保存指标重新生成报告")
    report_parser.add_argument("--run-dir", type=Path, required=True)

    retry_parser = subparsers.add_parser(
        "retry-rerank",
        help="复用共同候选，只重试失败的精排题",
    )
    retry_parser.add_argument("--run-dir", type=Path, required=True)
    retry_parser.add_argument(
        "--group",
        choices=("baseline", "candidate"),
        required=True,
    )
    retry_parser.add_argument("--concurrency", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "validate-dataset":
            return _validate_dataset(args.dataset)
        if args.command == "validate-experiment":
            return _validate_experiment(args.experiment)
        if args.command == "run":
            return _run(args)
        if args.command == "score":
            score_run_directory(
                args.run_dir.resolve(),
                show_progress=not args.no_progress,
                missing_only=args.missing_only,
            )
            print(f"评分完成：{args.run_dir.resolve()}")
            return 0
        if args.command == "report":
            run_dir = args.run_dir.resolve()
            reused = reconcile_run_identical_ragas_scores(run_dir)
            if reused:
                print(f"相同上下文复用评分：{reused} 道")
            comparison = generate_report(run_dir)
            print(f"报告完成：{args.run_dir.resolve() / 'report.md'}")
            print(f"结论：{comparison['verdict_text']}")
            return 0 if comparison["verdict"] != "evaluation_failed" else 2
        if args.command == "retry-rerank":
            retried = asyncio.run(
                retry_failed_shared_rerank(
                    run_dir=args.run_dir.resolve(),
                    group_name=args.group,
                    concurrency=args.concurrency,
                )
            )
            print(f"精排补跑完成：{retried} 道")
            return 0
    except (
        DatasetValidationError,
        ExperimentValidationError,
        EvaluationRunError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 1


def _validate_dataset(path: Path) -> int:
    dataset = load_dataset(path.resolve())
    basic = dataset.cases_for_suite("basic") if "basic" in dataset.suites else []
    counts = Counter(case.case_type for case in basic)
    print(f"数据集：{dataset.version}")
    print(f"知识库：{', '.join(dataset.knowledge_bases)}")
    print(f"题目总数：{len(dataset.cases)}")
    if basic:
        print(f"basic suite：{len(basic)} 道")
        print(
            "题型："
            f"direct_fact={counts['direct_fact']}，"
            f"paraphrase={counts['paraphrase']}，"
            f"multi_document={counts['multi_document']}"
        )
    print("结果：PASS")
    return 0


def _validate_experiment(path: Path) -> int:
    experiment = load_experiment(path.resolve(), project_root=PROJECT_ROOT)
    print(f"实验：{experiment.experiment_id}")
    print(f"数据集：{experiment.dataset}")
    print(f"suite：{experiment.suite}")
    print(f"baseline：{experiment.baseline.label}")
    print(f"candidate：{experiment.candidate.label}")
    print(f"唯一差异：{', '.join(experiment.changed_fields)}")
    print(f"主指标：{experiment.primary_metric}")
    print(
        "运行条件："
        f"业务并发={experiment.concurrency}，"
        f"RAGAS并发={experiment.ragas_concurrency}，"
        f"warm-up={experiment.warmup_cases}"
    )
    print("结果：PASS")
    return 0


def _run(args: argparse.Namespace) -> int:
    experiment_path = args.experiment.resolve()
    experiment = load_experiment(experiment_path, project_root=PROJECT_ROOT)
    dataset = load_dataset((PROJECT_ROOT / experiment.dataset).resolve())
    print(f"实验：{experiment.title}")
    print(f"题目：{experiment.suite} / {args.limit or len(dataset.cases_for_suite(experiment.suite))} 道")
    print(
        "索引："
        f"baseline={experiment.baseline.index_version}，"
        f"candidate={experiment.candidate.index_version}（请求中已显式锁定）"
    )
    print(f"业务请求并发：{experiment.concurrency}")
    print(f"RAGAS并发：{experiment.ragas_concurrency}")
    print("RAGAS telemetry: disabled")
    print("Offline Langfuse tracing: disabled")
    with _evaluation_api_key(args) as api_key:
        run_dir = asyncio.run(
            run_retrieval_groups(
                experiment=experiment,
                dataset=dataset,
                api_key=api_key,
                base_url=args.base_url,
                output_root=args.output_root.resolve(),
                limit=args.limit,
                case_ids=args.case_ids,
                output_suffix=args.output_suffix,
            )
        )
    print(f"检索完成：{run_dir}")
    if experiment.baseline_run_dir:
        source_dir = Path(experiment.baseline_run_dir)
        if not source_dir.is_absolute():
            source_dir = PROJECT_ROOT / source_dir
        source_metrics = source_dir / f"{experiment.baseline_group}.metrics.json"
        if not source_metrics.is_file():
            raise EvaluationRunError(f"冻结 baseline metrics 不存在: {source_metrics}")
        shutil.copyfile(source_metrics, run_dir / "baseline.metrics.json")
    score_run_directory(
        run_dir,
        show_progress=not args.no_progress,
        missing_only=bool(experiment.baseline_run_dir),
    )
    print("评分完成")
    comparison = generate_report(run_dir)
    print(f"报告：{run_dir / 'report.md'}")
    print(f"结论：{comparison['verdict_text']}")
    print(
        json.dumps(
            {
                "primary_metric": comparison["primary_metric"],
                "delta": comparison["delta"].get(comparison["primary_metric"]),
                "p95_delta_ms": comparison["delta"].get("latency_p95_ms"),
                "model_calls_delta": comparison["delta"].get(
                    "average_application_model_calls"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if comparison["verdict"] != "evaluation_failed" else 2


@contextmanager
def _evaluation_api_key(args: argparse.Namespace) -> Iterator[str]:
    configured_key = args.api_key or os.getenv("RAG_EVAL_API_KEY")
    if configured_key:
        yield configured_key
        return
    if not args.tenant_id:
        raise EvaluationRunError(
            "缺少评测身份：请传 --api-key、设置 RAG_EVAL_API_KEY，"
            "或在本地使用 --tenant-id 创建临时 Key"
        )

    from app.db.session import SessionLocal
    from app.models.api_key import ApiKey
    from app.repositories.api_key_repository import ApiKeyRepository
    from app.repositories.tenant_repository import TenantRepository
    from app.utils.api_key_crypto import generate_api_key_material
    from app.utils.id_generator import generate_uuid
    from app.utils.tenant_status import ApiKeyStatus

    db = SessionLocal()
    key_id = generate_uuid()
    key_created = False
    try:
        tenant = TenantRepository(db).get_by_id(args.tenant_id)
        if tenant is None:
            raise EvaluationRunError(f"评测租户不存在: {args.tenant_id}")
        raw_key, key_hash, key_prefix = generate_api_key_material()
        ApiKeyRepository(db).create(
            ApiKey(
                id=key_id,
                tenant_id=args.tenant_id,
                key_hash=key_hash,
                key_prefix=key_prefix,
                name="unified-eval-temporary",
                status=ApiKeyStatus.ACTIVE,
                expires_at=None,
            )
        )
        db.commit()
        key_created = True
        yield raw_key
    finally:
        try:
            if key_created:
                api_key = db.get(ApiKey, key_id)
                if api_key is not None:
                    db.delete(api_key)
                    db.commit()
        finally:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
