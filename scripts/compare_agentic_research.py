from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.index_version import KnowledgeBaseIndexVersion
from app.models.knowledge_base import KnowledgeBase
from app.models.tenant import Tenant
from app.schemas.rag import EvidenceOptions, QueryOptions, RerankOptions, RetrieveRequest
from app.schemas.research import ResearchRequest
from app.services.rag_service import RagService
from app.services.research_decider_service import RESEARCH_DECIDER_PROMPT
from app.services.research_finalizer_service import RESEARCH_FINALIZER_PROMPT
from app.services.research_planner_service import RESEARCH_PLANNER_PROMPT
from app.services.research_service import ResearchService

Mode = Literal["single", "decomposition", "research"]
TENANT_ID = "tenant_ecom_pro"
KB_ID = "df1dad54-7bfd-4ec7-9116-cfe2448e9389"
HR_TRAVEL_KB_ID = "524146da-ad39-4c93-bee0-12b299d63af6"
INDEX_VERSION = "v2"
CASE_TIMEOUT_SECONDS = 90
CASE_TYPES = (
    "simple",
    "independent_multi_aspect",
    "dependent_multi_hop",
    "control",
)


def aspect(aspect_id: str, description: str, *chunk_ids: str) -> dict[str, Any]:
    return {
        "aspect_id": aspect_id,
        "description": description,
        "acceptable_chunk_ids": list(chunk_ids),
    }


CASES: list[dict[str, Any]] = [
    {
        "id": "payment_duplicate_charge",
        "case_type": "simple",
        "question": "同一个订单被重复扣款，多余的钱多久退回？",
        "ground_truth": "系统识别重复扣款后自动退款，通常1至3个工作日。",
        "required_aspects": [
            aspect(
                "A1",
                "重复扣款的处理方式和到账时间",
                "e3bd1467-f992-48f9-9483-47f14ff49c4c",
            )
        ],
        "expected_documents": ["9b4bac02-5e6f-4d50-b6e2-83dde5ac5269"],
        "expected_status": "complete",
    },
    {
        "id": "member_plus_cancel",
        "case_type": "simple",
        "question": "Plus年卡买后7天内可以全额退款吗？",
        "ground_truth": "7日内且未使用专属权益可全额退；已使用则按剩余天数折算。",
        "required_aspects": [
            aspect(
                "A1",
                "Plus年卡全额退款条件",
                "d143f7a0-889b-448c-b446-6314cf970dc7",
            )
        ],
        "expected_documents": ["185b2ef5-15f7-4983-95ef-4a5a1875ae0a"],
        "expected_status": "complete",
    },
    {
        "id": "live_order_responsibility",
        "case_type": "simple",
        "question": "直播间下单后由主播还是店铺商家承担售后和发票义务？",
        "ground_truth": "售后和发票义务主体是店铺商家，达人通常不是合同卖方。",
        "required_aspects": [
            aspect(
                "A1",
                "直播订单售后和发票的责任主体",
                "c1898591-8556-45fc-a8f5-14b24dc525ca",
                "1ed0393c-580a-4a3f-b98c-ead0d0a9e010",
            )
        ],
        "expected_documents": ["bdf47657-fec9-4112-a059-08abc6431696"],
        "expected_status": "complete",
    },
    {
        "id": "search_personalization_off",
        "case_type": "simple",
        "question": "关闭个性化推荐后是不是就完全看不到推荐和广告了？",
        "ground_truth": "不会，仍展示热门或编辑精选，广告也不保证完全消失。",
        "required_aspects": [
            aspect(
                "A1",
                "关闭个性化后的推荐和广告表现",
                "ab9b5fc7-dde5-4d38-a5c6-92d34d57d90c",
            )
        ],
        "expected_documents": ["872cc351-513b-490f-ac35-c5485de799b1"],
        "expected_status": "complete",
    },
    {
        "id": "payment_combined_refund",
        "case_type": "independent_multi_aspect",
        "question": "组合支付发生部分退款后，银行卡、优惠券和积分分别怎么退？",
        "ground_truth": "现金按原支付比例原路退，券按活动规则处理，积分按会员规则退回。",
        "required_aspects": [
            aspect("A1", "银行卡退款路径", "2af0bac1-6126-4d91-bc42-774ca90d9d4d"),
            aspect("A2", "优惠券退款规则", "68a004e5-c525-4ae7-8613-759d0568d893"),
            aspect("A3", "积分退款规则", "779b77ca-dcb0-4de1-9eaf-d2279a876f34"),
        ],
        "expected_documents": [
            "9b4bac02-5e6f-4d50-b6e2-83dde5ac5269",
            "8b484755-a80d-44a2-85c0-0b21e53247bc",
            "185b2ef5-15f7-4983-95ef-4a5a1875ae0a",
        ],
        "expected_status": "complete",
    },
    {
        "id": "payment_risk_appeal",
        "case_type": "independent_multi_aspect",
        "question": "异地登录后大额支付被风控拦截，怎么申诉，多久复核？",
        "ground_truth": "先完成二次验证；误杀可提交支付安全申诉，48小时内复核。",
        "required_aspects": [
            aspect("A1", "异常大额支付的验证方式", "aa81a2e2-a349-4946-8cbd-d3bdd40f078f"),
            aspect("A2", "风控误杀申诉和复核时限", "aa81a2e2-a349-4946-8cbd-d3bdd40f078f"),
        ],
        "expected_documents": ["9b4bac02-5e6f-4d50-b6e2-83dde5ac5269"],
        "expected_status": "complete",
    },
    {
        "id": "member_points_refund",
        "case_type": "independent_multi_aspect",
        "question": "积分抵扣订单退款后，积分和现金分别怎么退？",
        "ground_truth": "积分等值退回积分池，过期不退；现金按原支付渠道退回。",
        "required_aspects": [
            aspect("A1", "积分退款及过期处理", "779b77ca-dcb0-4de1-9eaf-d2279a876f34"),
            aspect("A2", "现金原支付渠道退回", "2af0bac1-6126-4d91-bc42-774ca90d9d4d"),
        ],
        "expected_documents": [
            "185b2ef5-15f7-4983-95ef-4a5a1875ae0a",
            "9b4bac02-5e6f-4d50-b6e2-83dde5ac5269",
        ],
        "expected_status": "complete",
    },
    {
        "id": "aftersales_seven_day_freight",
        "case_type": "independent_multi_aspect",
        "question": "七天无理由退货要满足哪些条件，退货运费由谁出？",
        "ground_truth": "7个自然日内发起且商品完好；无理由退货运费通常由买家承担。",
        "required_aspects": [
            aspect("A1", "七天无理由退货条件", "8e274390-4b66-4695-a946-f86570388c01"),
            aspect("A2", "无理由退货运费承担方", "dd35b70f-2716-4b45-907d-e9b93562f0fa"),
        ],
        "expected_documents": ["71785118-9dcb-41cc-97bf-4e8dbf83c27b"],
        "expected_status": "complete",
    },
    {
        "id": "live_category_qualification",
        "case_type": "dependent_multi_hop",
        "question": "直播带货美妆或医疗器械需要哪些额外资质？",
        "ground_truth": "达人先取得带货权限，并满足对应类目资质要求。",
        "required_aspects": [
            aspect("A1", "达人直播带货权限", "6400b836-5cf1-43fc-89b8-a2f57e5629d2"),
            aspect("A2", "美妆类目具体资质", "44159d28-676b-40c1-a84f-34b8e4ac6c18"),
        ],
        "expected_documents": [
            "bdf47657-fec9-4112-a059-08abc6431696",
            "4b932e57-a65e-46c8-b8c9-fa9637fd40cd",
        ],
        "expected_status": "complete",
        "dependency": {
            "discover_from_chunk_ids": ["6400b836-5cf1-43fc-89b8-a2f57e5629d2"],
            "target_chunk_ids": ["44159d28-676b-40c1-a84f-34b8e4ac6c18"],
        },
    },
    {
        "id": "live_campaign_shipping",
        "case_type": "dependent_multi_hop",
        "question": "大场直播产生的大促订单多久要发货，超时如何处理？",
        "ground_truth": "适用活动SLA；通常48小时内揽收，超时按活动规则赔付。",
        "required_aspects": [
            aspect("A1", "直播大场订单适用活动规则", "c4c6be7a-ef36-4a1d-99e9-eff07b442873"),
            aspect(
                "A2",
                "大促订单48小时揽收",
                "413ec4ca-4474-4c54-a97e-ee642182aa66",
                "be592533-2534-499a-ad02-ed0888908d52",
            ),
            aspect("A3", "延迟发货赔付规则", "be592533-2534-499a-ad02-ed0888908d52"),
        ],
        "expected_documents": [
            "bdf47657-fec9-4112-a059-08abc6431696",
            "ba5cb3bc-e642-4d39-926f-e9736d80d6c4",
            "8b484755-a80d-44a2-85c0-0b21e53247bc",
        ],
        "expected_status": "complete",
        "dependency": {
            "discover_from_chunk_ids": ["c4c6be7a-ef36-4a1d-99e9-eff07b442873"],
            "target_chunk_ids": [
                "413ec4ca-4474-4c54-a97e-ee642182aa66",
                "be592533-2534-499a-ad02-ed0888908d52",
            ],
        },
    },
    {
        "id": "search_ad_invoice_reconciliation",
        "case_type": "dependent_multi_hop",
        "question": "广告费账单有差异怎么申诉，广告费发票由谁开？",
        "ground_truth": "账单异议7天内提出；广告费发票由平台向商家开具。",
        "required_aspects": [
            aspect("A1", "广告账单差异申诉期限", "079529f9-bae4-48d6-8917-8d3582ed77da"),
            aspect("A2", "平台开具广告费发票", "cb269192-7e36-4a31-9832-174fee8ad306"),
            aspect("A3", "商家结算对账入口与期限", "da6c6060-b3d8-4966-bc72-129292455044"),
        ],
        "expected_documents": [
            "872cc351-513b-490f-ac35-c5485de799b1",
            "d3049f5c-5617-4fe0-98fb-e18a37d45cb7",
        ],
        "expected_status": "complete",
        "dependency": {
            "discover_from_chunk_ids": ["079529f9-bae4-48d6-8917-8d3582ed77da"],
            "target_chunk_ids": [
                "cb269192-7e36-4a31-9832-174fee8ad306",
                "da6c6060-b3d8-4966-bc72-129292455044",
            ],
        },
    },
    {
        "id": "aftersales_intercept_refund",
        "case_type": "dependent_multi_hop",
        "question": "商品已经发货但未签收，拦截成功后怎么退款，失败怎么办？",
        "ground_truth": "拦截成功则退款；失败走退货流程，入库质检后触发退款。",
        "required_aspects": [
            aspect("A1", "拦截成功和失败后的处理", "73703d96-4fda-4cab-a2da-7be08461020b"),
            aspect("A2", "退货质检和退款触发", "44d9f1c9-99b3-47b0-8a89-927da0bbb9c0"),
        ],
        "expected_documents": [
            "71785118-9dcb-41cc-97bf-4e8dbf83c27b",
            "ba5cb3bc-e642-4d39-926f-e9736d80d6c4",
        ],
        "expected_status": "complete",
        "dependency": {
            "discover_from_chunk_ids": ["73703d96-4fda-4cab-a2da-7be08461020b"],
            "target_chunk_ids": ["44d9f1c9-99b3-47b0-8a89-927da0bbb9c0"],
        },
    },
    {
        "id": "control_no_answer_offline_parking",
        "case_type": "control",
        "question": "线下门店停车费可以用平台积分抵扣吗？",
        "ground_truth": "当前知识库没有线下门店停车费规则，应明确材料不足。",
        "required_aspects": [aspect("A1", "线下门店停车费积分抵扣规则")],
        "expected_documents": [],
        "expected_status": "missing",
        "control": {"kind": "no_answer"},
    },
    {
        "id": "control_partial_plus_family",
        "case_type": "control",
        "question": "Plus年卡7天内怎么退，家庭成员共享后还能全额退吗？",
        "ground_truth": "年卡退款规则存在，家庭成员共享后的处理规则不存在。",
        "required_aspects": [
            aspect("A1", "Plus年卡退款规则", "d143f7a0-889b-448c-b446-6314cf970dc7"),
            aspect("A2", "家庭成员共享后的退款规则"),
        ],
        "expected_documents": ["185b2ef5-15f7-4983-95ef-4a5a1875ae0a"],
        "expected_status": "partial",
        "control": {"kind": "partial_answer"},
    },
    {
        "id": "control_shipping_scope_conflict",
        "case_type": "control",
        "question": "普通现货订单和大促订单分别多久揽收，两个时限冲突吗？",
        "ground_truth": "普通现货24小时、大促48小时，适用范围不同，不应混为冲突。",
        "required_aspects": [
            aspect("A1", "普通现货24小时揽收", "413ec4ca-4474-4c54-a97e-ee642182aa66"),
            aspect(
                "A2",
                "大促订单48小时揽收及范围",
                "413ec4ca-4474-4c54-a97e-ee642182aa66",
                "be592533-2534-499a-ad02-ed0888908d52",
            ),
        ],
        "expected_documents": [
            "ba5cb3bc-e642-4d39-926f-e9736d80d6c4",
            "8b484755-a80d-44a2-85c0-0b21e53247bc",
        ],
        "expected_status": "complete",
        "control": {"kind": "scoped_conflict"},
    },
    {
        "id": "control_permission_version_boundary",
        "case_type": "control",
        "question": "候选人背调一般需要几天，必须先取得什么授权？",
        "ground_truth": "答案只存在于其他租户v1索引，本租户v2知识库必须返回材料不足。",
        "required_aspects": [
            aspect("A1", "候选人背调时长"),
            aspect("A2", "候选人背调授权要求"),
        ],
        "expected_documents": [],
        "expected_status": "missing",
        "control": {
            "kind": "permission_and_version_boundary",
            "forbidden_chunk_ids": [
                "4a7e109e-1a22-4c4c-adc7-d6c0d12fa366",
                "7721e4f6-9639-4104-a20b-dfcb26ced6e6",
            ],
        },
    },
]


DEPENDENT_SMOKE_CASES: list[dict[str, Any]] = [
    {
        "id": "probation_senior_engineer_beijing",
        "case_type": "dependent_multi_hop",
        "question": "一名试用期资深工程师去北京出差3天，预计费用8500元，可以乘坐什么交通工具，酒店每晚上限多少，最低需要哪些人审批？",
        "ground_truth": (
            "资深工程师通常为P6，对应T2；北京为A+类。T2可乘高铁二等座、普通列车硬卧或软卧，"
            "铁路最快直达超过4小时可选经济舱；住宿上限650元/晚。3天且8500元常规最低由直属负责人审批，"
            "试用期还必须增加部门负责人审批。"
        ),
        "required_aspects": [
            aspect("A1", "资深工程师对应T2", "1bb20f02-8530-4ac7-983a-aca881ee9afa"),
            aspect("A2", "T2可用的国内城际交通标准", "7caa5444-22fe-418d-8d06-0a29d4238dcc"),
            aspect("A3", "T2在北京的住宿上限为650元", "c502f223-22f7-4e2d-b367-30b9059a1308"),
            aspect("A4", "3天8500元及试用期的最低审批要求", "0b52462e-6429-4741-be9e-5c4ea0bcea75", "7d4146cf-c9e4-4e91-9c65-7c00eb29c916"),
        ],
        "expected_documents": [
            "705ba5dc-2302-4981-a9de-1fb9fbe2161b",
            "b3815cd9-e636-485d-9250-2e6372e246c0",
            "83df86d5-b9f3-49b4-bad2-effe4d35e572",
            "b0b9273c-5faf-4d87-aa6a-bd04556df045",
        ],
        "expected_status": "complete",
        "dependency": {
            "discover_from_chunk_ids": ["dde882be-0bf5-4cf2-9202-c3b57ab978c1"],
            "intermediate_chunk_ids": [
                "1bb20f02-8530-4ac7-983a-aca881ee9afa",
                "7288f730-ea3c-463b-8037-67ad46a6b1ef",
            ],
            "target_chunk_ids": [
                "7caa5444-22fe-418d-8d06-0a29d4238dcc",
                "c502f223-22f7-4e2d-b367-30b9059a1308",
                "0b52462e-6429-4741-be9e-5c4ea0bcea75",
                "7d4146cf-c9e4-4e91-9c65-7c00eb29c916",
            ],
        },
    },
    {
        "id": "intern_wuhan_emergency_solo_trip",
        "case_type": "dependent_multi_hop",
        "question": "一名实习生需要单独去武汉处理2天生产故障，这种情况是否允许，交通和住宿按什么标准，来不及事前走完审批该怎么办？",
        "ground_truth": (
            "实习生统一按T0，原则上不得单独出差；确需单独出差须HRBP和部门负责人共同批准。"
            "武汉为B类，T0可乘高铁二等座、普通列车硬座或硬卧，原则上不乘飞机；经批准可乘经济舱，"
            "住宿上限380元/晚。生产故障可先取得直属负责人即时确认后出发，并在1个工作日内补齐审批。"
        ),
        "required_aspects": [
            aspect("A1", "实习生统一按T0", "29f3de37-0667-4c7b-bf10-342d3b869a0f", "b211a933-380c-44d1-8859-c6592d897897"),
            aspect("A2", "实习生单独出差的限制与审批", "b211a933-380c-44d1-8859-c6592d897897", "7d4146cf-c9e4-4e91-9c65-7c00eb29c916"),
            aspect("A3", "T0国内城际交通标准", "7caa5444-22fe-418d-8d06-0a29d4238dcc"),
            aspect("A4", "T0在武汉的住宿上限为380元", "c502f223-22f7-4e2d-b367-30b9059a1308"),
            aspect("A5", "生产故障紧急出差的确认与补批时限", "62c4a19b-d6a9-42dd-b3e9-adf0fd869e30", "94719d4a-f0cf-4ee6-ab54-36b97c225e39"),
        ],
        "expected_documents": [
            "77e778b0-d96a-4d30-9b5d-3879639cbf60",
            "e8e7154e-b7bb-4b2f-84d6-8d6cf96f9d05",
            "b3815cd9-e636-485d-9250-2e6372e246c0",
            "83df86d5-b9f3-49b4-bad2-effe4d35e572",
            "b0b9273c-5faf-4d87-aa6a-bd04556df045",
        ],
        "expected_status": "complete",
        "dependency": {
            "discover_from_chunk_ids": ["29f3de37-0667-4c7b-bf10-342d3b869a0f"],
            "intermediate_chunk_ids": [
                "1bb20f02-8530-4ac7-983a-aca881ee9afa",
                "d1d7a092-e6ad-4350-ae49-4ee6d924c7af",
            ],
            "target_chunk_ids": [
                "b211a933-380c-44d1-8859-c6592d897897",
                "7caa5444-22fe-418d-8d06-0a29d4238dcc",
                "c502f223-22f7-4e2d-b367-30b9059a1308",
                "62c4a19b-d6a9-42dd-b3e9-adf0fd869e30",
            ],
        },
    },
    {
        "id": "research_manager_suzhou_late_arrival",
        "case_type": "dependent_multi_hop",
        "question": "一名研发经理去苏州工业园区出差，晚上23:30到站，他的差旅等级是什么，酒店每晚上限多少，从车站打车到酒店的单程上限多少？",
        "ground_truth": "研发经理通常对应M2，映射为T2；苏州工业园区按A类。T2在A类城市住宿上限600元/晚；23:30属于深夜到达，A类城市车站到住宿地的必要接驳单程不超过250元。",
        "required_aspects": [
            aspect("A1", "研发经理M2对应T2", "1bb20f02-8530-4ac7-983a-aca881ee9afa"),
            aspect("A2", "T2在苏州工业园区住宿上限600元", "c502f223-22f7-4e2d-b367-30b9059a1308"),
            aspect("A3", "A类城市深夜车站接驳单程上限250元", "00c23298-261b-4c1e-b468-127dc91b16c5"),
        ],
        "expected_documents": [
            "705ba5dc-2302-4981-a9de-1fb9fbe2161b",
            "83df86d5-b9f3-49b4-bad2-effe4d35e572",
            "e8e7154e-b7bb-4b2f-84d6-8d6cf96f9d05",
        ],
        "expected_status": "complete",
        "dependency": {
            "discover_from_chunk_ids": ["51830cf8-0700-42f6-9797-e920a75b1d31"],
            "intermediate_chunk_ids": [
                "1bb20f02-8530-4ac7-983a-aca881ee9afa",
                "2cfb8c24-de2f-4e97-b99b-3e73a08f8f63",
            ],
            "target_chunk_ids": [
                "c502f223-22f7-4e2d-b367-30b9059a1308",
                "00c23298-261b-4c1e-b468-127dc91b16c5",
            ],
        },
    },
    {
        "id": "technical_expert_singapore_long_flight",
        "case_type": "dependent_multi_hop",
        "question": "一名技术专家去新加坡出差，单段计划飞行9小时，他的差旅等级是什么，最低需要哪些人审批，可以坐什么舱位，酒店每晚上限多少？",
        "ground_truth": (
            "技术专家通常对应P7，映射为T3。新加坡属于R1和G1；最低由部门负责人、国际业务负责人审批。"
            "T3单段飞行超过8小时可乘超级经济舱，G1住宿上限为1800元/晚。"
        ),
        "required_aspects": [
            aspect("A1", "技术专家P7对应T3", "1bb20f02-8530-4ac7-983a-aca881ee9afa"),
            aspect("A2", "新加坡属于R1常规风险地区", "1dac2058-f6f9-4ecd-bd35-9702b0fc5a02"),
            aspect("A3", "R1国际出差的最低审批人", "ae7897a0-2018-42fb-aef9-6690e8be90ef"),
            aspect("A4", "T3单段飞行超过8小时可乘超级经济舱", "9c2272d9-4edc-4aa7-a4a6-2f773f2110d4"),
            aspect("A5", "T3在新加坡G1地区住宿上限1800元", "6864207d-325b-4180-b065-8487993136d2"),
        ],
        "expected_documents": [
            "705ba5dc-2302-4981-a9de-1fb9fbe2161b",
            "f3bbb3a9-dbde-4a6d-a189-ec7aa7b48f91",
        ],
        "expected_status": "complete",
        "dependency": {
            "discover_from_chunk_ids": ["dde882be-0bf5-4cf2-9202-c3b57ab978c1"],
            "intermediate_chunk_ids": [
                "1bb20f02-8530-4ac7-983a-aca881ee9afa",
                "1dac2058-f6f9-4ecd-bd35-9702b0fc5a02",
            ],
            "target_chunk_ids": [
                "ae7897a0-2018-42fb-aef9-6690e8be90ef",
                "9c2272d9-4edc-4aa7-a4a6-2f773f2110d4",
                "6864207d-325b-4180-b065-8487993136d2",
            ],
        },
    },
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def prompt_hashes() -> dict[str, str]:
    return {
        "planner": sha256_text(RESEARCH_PLANNER_PROMPT),
        "decider": sha256_text(RESEARCH_DECIDER_PROMPT),
        "finalizer": sha256_text(RESEARCH_FINALIZER_PROMPT),
    }


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "app").rglob("*.py"))
    paths.append(Path(__file__))
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def freeze_cases(
    sources: list[dict[str, Any]],
    *,
    kb_id: str = KB_ID,
    index_version: str = INDEX_VERSION,
) -> list[dict[str, Any]]:
    cases = []
    for source in sources:
        item = {
            **source,
            "kb_ids": [kb_id],
            "index_versions": {kb_id: index_version},
            "requires_multi_hop": source["case_type"] == "dependent_multi_hop",
            "dependency": source.get("dependency"),
            "review_status": "reviewed",
        }
        cases.append(item)
    return cases


def annotation_sha256(cases: list[dict[str, Any]]) -> str:
    annotation_payload = [
        {
            "id": item["id"],
            "required_aspects": item["required_aspects"],
            "expected_status": item["expected_status"],
            "dependency": item.get("dependency"),
            "control": item.get("control"),
        }
        for item in cases
    ]
    return sha256_text(json.dumps(annotation_payload, ensure_ascii=False, sort_keys=True))


def prepare_dataset() -> dict[str, Any]:
    cases = freeze_cases(CASES)
    return {
        "name": "opt18_controlled_agentic_research_16",
        "evaluation_scope": "formal",
        "created_at": utc_now(),
        "tenant_id": TENANT_ID,
        "profile": "quality",
        "top_k": 10,
        "annotation": {
            "status": "reviewed",
            "method": "document_grounded_manual_review_reusing_opt16_opt17_frozen_chunks",
            "annotator": "project_owner",
            "reviewer": "optimization_18_implementation_review",
            "note": (
                "12个质量题复用优化十六、十七已人工复核的真实Chunk；"
                "4个控制题按当前数据库中的授权范围和索引版本复核。"
            ),
        },
        "annotation_sha256": annotation_sha256(cases),
        "cases": cases,
    }


def prepare_smoke_dataset() -> dict[str, Any]:
    cases = freeze_cases(DEPENDENT_SMOKE_CASES, kb_id=HR_TRAVEL_KB_ID)
    return {
        "name": "opt18_hr_travel_dependent_multi_hop_v1",
        "evaluation_scope": "smoke",
        "created_at": utc_now(),
        "tenant_id": TENANT_ID,
        "profile": "quality",
        "top_k": 10,
        "annotation": {
            "status": "reviewed",
            "method": "document_grounded_hr_travel_policy_review",
            "annotator": "project_owner",
            "reviewer": "optimization_18_dependency_chain_review",
            "note": (
                "四题覆盖人员身份、岗位职级、差旅等级、城市或风险分类及最终费用审批规则；"
                "所有证据均绑定人事与差旅知识库的真实v2 Chunk。"
            ),
        },
        "annotation_sha256": annotation_sha256(cases),
        "cases": cases,
    }


def validate_shape(dataset: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        return ["cases must be an array"]
    counts = Counter(str(case.get("case_type")) for case in cases)
    scope = dataset.get("evaluation_scope", "formal")
    if scope == "formal":
        if len(cases) != 16:
            errors.append(f"case count must be 16, got {len(cases)}")
        for case_type in CASE_TYPES:
            if counts[case_type] != 4:
                errors.append(f"{case_type} count must be 4, got {counts[case_type]}")
    elif scope == "smoke":
        if len(cases) != 4 or counts != Counter({"dependent_multi_hop": 4}):
            errors.append("smoke dataset must contain 4 dependent_multi_hop cases")
    else:
        errors.append(f"unknown evaluation_scope: {scope}")
    ids = [str(case.get("id") or "") for case in cases]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        errors.append("case ids must be non-empty and unique")
    if dataset.get("tenant_id") != TENANT_ID:
        errors.append(f"tenant_id must be {TENANT_ID}")
    if (dataset.get("annotation") or {}).get("status") != "reviewed":
        errors.append("dataset annotation must be reviewed")
    for case in cases:
        case_id = str(case.get("id") or "<unknown>")
        if case.get("review_status") != "reviewed":
            errors.append(f"{case_id}: review_status must be reviewed")
        kb_ids = case.get("kb_ids") or []
        versions = case.get("index_versions") or {}
        if len(kb_ids) != 1:
            errors.append(f"{case_id}: exactly one kb_id must be frozen")
        elif versions != {kb_ids[0]: INDEX_VERSION}:
            errors.append(f"{case_id}: index version must be frozen to {INDEX_VERSION}")
        aspects = case.get("required_aspects") or []
        aspect_ids = [str(value.get("aspect_id") or "") for value in aspects]
        if not aspects or len(aspect_ids) != len(set(aspect_ids)) or any(not x for x in aspect_ids):
            errors.append(f"{case_id}: required aspects must be non-empty and unique")
        expected_status = case.get("expected_status")
        has_any_evidence = any(value.get("acceptable_chunk_ids") for value in aspects)
        all_have_evidence = all(value.get("acceptable_chunk_ids") for value in aspects)
        derived_status = "complete" if all_have_evidence else "partial" if has_any_evidence else "missing"
        if expected_status != derived_status:
            errors.append(
                f"{case_id}: expected_status {expected_status} != annotation {derived_status}"
            )
        dependency = case.get("dependency")
        if case.get("requires_multi_hop"):
            discover = set((dependency or {}).get("discover_from_chunk_ids") or [])
            intermediate = set((dependency or {}).get("intermediate_chunk_ids") or [])
            target = set((dependency or {}).get("target_chunk_ids") or [])
            if not discover or not target or discover & target:
                errors.append(f"{case_id}: invalid dependency annotation")
            if scope == "smoke" and (
                not intermediate
                or discover & intermediate
                or intermediate & target
            ):
                errors.append(f"{case_id}: smoke case needs a distinct intermediate path")
        elif dependency is not None:
            errors.append(f"{case_id}: non multi-hop case has dependency")
    return errors


def validate_database(dataset: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    with SessionLocal() as db:
        tenant = db.get(Tenant, dataset["tenant_id"])
        if tenant is None or tenant.plan != "pro" or tenant.status != "active":
            errors.append("evaluation tenant must exist, be active, and use pro plan")
        scopes = {
            (case["kb_ids"][0], case["index_versions"][case["kb_ids"][0]])
            for case in dataset["cases"]
        }
        for kb_id, index_version in scopes:
            kb = db.get(KnowledgeBase, kb_id)
            if kb is None or kb.tenant_id != TENANT_ID:
                errors.append(f"evaluation knowledge base is outside the tenant: {kb_id}")
                continue
            version = db.scalar(
                select(KnowledgeBaseIndexVersion).where(
                    KnowledgeBaseIndexVersion.tenant_id == TENANT_ID,
                    KnowledgeBaseIndexVersion.kb_id == kb_id,
                    KnowledgeBaseIndexVersion.version == index_version,
                )
            )
            if version is None or version.status not in {"ready", "active", "archived"}:
                errors.append(f"evaluation index version is not queryable: {kb_id}/{index_version}")

        for case in dataset["cases"]:
            kb_id = case["kb_ids"][0]
            index_version = case["index_versions"][kb_id]
            annotated_ids = {
                str(chunk_id)
                for value in case["required_aspects"]
                for chunk_id in value.get("acceptable_chunk_ids") or []
            }
            annotated_ids.update(
                str(chunk_id)
                for key in (
                    "discover_from_chunk_ids",
                    "intermediate_chunk_ids",
                    "target_chunk_ids",
                )
                for chunk_id in (case.get("dependency") or {}).get(key) or []
            )
            in_scope = {
                chunk.id: chunk
                for chunk in db.scalars(
                    select(Chunk).where(
                        Chunk.id.in_(annotated_ids),
                        Chunk.tenant_id == TENANT_ID,
                        Chunk.kb_id == kb_id,
                        Chunk.index_version == index_version,
                    )
                ).all()
            }
            missing = sorted(annotated_ids - set(in_scope))
            if missing:
                errors.append(f"{case['id']}: annotated chunks missing: {missing}")
            expected_docs = set(case.get("expected_documents") or [])
            actual_docs = {chunk.document_id for chunk in in_scope.values()}
            missing_docs = sorted(expected_docs - actual_docs)
            if missing_docs:
                errors.append(f"{case['id']}: expected documents missing: {missing_docs}")

            forbidden_ids = set((case.get("control") or {}).get("forbidden_chunk_ids") or [])
            forbidden = {
                chunk.id: chunk
                for chunk in db.scalars(select(Chunk).where(Chunk.id.in_(forbidden_ids))).all()
            }
            if set(forbidden) != forbidden_ids:
                errors.append(f"{case['id']}: forbidden boundary chunks must exist")
            escaped = [
                chunk_id
                for chunk_id, chunk in forbidden.items()
                if chunk.tenant_id == TENANT_ID
                and chunk.kb_id == kb_id
                and chunk.index_version == index_version
            ]
            if escaped:
                errors.append(f"{case['id']}: forbidden chunks escaped scope: {escaped}")
    return errors


def validate_dataset(dataset: dict[str, Any], *, check_database: bool = True) -> dict[str, Any]:
    errors = validate_shape(dataset)
    if check_database and not errors:
        try:
            errors.extend(validate_database(dataset))
        except Exception as exc:  # noqa: BLE001 - CLI must report unavailable dependencies.
            errors.append(f"database validation failed: {type(exc).__name__}: {str(exc)[:200]}")
    return {
        "valid": not errors,
        "case_count": len(dataset.get("cases") or []),
        "type_counts": dict(Counter(case.get("case_type") for case in dataset.get("cases") or [])),
        "errors": errors,
    }


def endpoint_available(url: str) -> bool:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def run_metadata(dataset_path: Path, mode: Mode) -> dict[str, Any]:
    dataset = load_json(dataset_path)
    kb_ids = sorted({kb_id for case in dataset["cases"] for kb_id in case["kb_ids"]})
    index_versions = {
        kb_id: version
        for case in dataset["cases"]
        for kb_id, version in case["index_versions"].items()
    }
    return {
        "run_id": os.urandom(8).hex(),
        "run_started_at": utc_now(),
        "dataset_sha256": sha256_file(dataset_path),
        "code_commit": git_value("rev-parse", "HEAD"),
        "code_dirty": bool(git_value("status", "--porcelain")),
        "code_sha256": code_fingerprint(),
        "mode": mode,
        "provider": settings.llm_provider,
        "models": {"fast": settings.llm_model, "strong": settings.llm_strong_model},
        "prompt_sha256": prompt_hashes(),
        "tenant_id": TENANT_ID,
        "profile": "quality",
        "kb_ids": kb_ids,
        "index_versions": index_versions,
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
        "repeat": 1,
        "warmup": 0,
    }


class OutputLock:
    def __init__(self, _path: Path) -> None:
        # One global lock prevents A/B/C commands from being launched concurrently,
        # even though they write to different result files.
        self.path = PROJECT_ROOT / "eval/opt18_research/.run.lock"
        self.fd: int | None = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(self.fd, f"pid={os.getpid()}\n".encode())

    def __exit__(self, *_args: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def balanced_cases(cases: list[dict[str, Any]], maximum: int | None) -> list[dict[str, Any]]:
    if maximum is None or maximum >= len(cases):
        return cases
    grouped = {case_type: [] for case_type in CASE_TYPES}
    for case in cases:
        grouped[case["case_type"]].append(case)
    ordered: list[dict[str, Any]] = []
    for index in range(4):
        for case_type in CASE_TYPES:
            if index < len(grouped[case_type]):
                ordered.append(grouped[case_type][index])
    return ordered[:maximum]


def select_cases(
    cases: list[dict[str, Any]],
    maximum: int | None,
    case_types: list[str] | None,
) -> list[dict[str, Any]]:
    if not case_types:
        return balanced_cases(cases, maximum)
    allowed = set(case_types)
    selected = [case for case in cases if case["case_type"] in allowed]
    return selected if maximum is None else selected[:maximum]


def pack_from_data(data: dict[str, Any]) -> dict[str, Any] | None:
    value = data.get("evidence_pack")
    return value if isinstance(value, dict) else None


async def execute_case(mode: Mode, case: dict[str, Any]) -> dict[str, Any]:
    kb_id = case["kb_ids"][0]
    index_version = case["index_versions"][kb_id]
    with SessionLocal() as db:
        if mode == "single":
            service = RagService(db)
            response = await service.retrieve(
                TENANT_ID,
                RetrieveRequest(
                    kb_ids=case["kb_ids"],
                    user_id="opt18-acceptance",
                    query=case["question"],
                    index_version=index_version,
                    profile="quality",
                    top_k=10,
                    rerank_options=RerankOptions(enabled=True, top_n=10),
                    evidence_options=EvidenceOptions(enabled=True, max_items=8),
                    query_options=QueryOptions(enabled=False, strategy="noop"),
                ),
            )
            data = response.model_dump(mode="json")
            evidence_usage = getattr(
                service.evidence_orchestration_service.llm_provider,
                "last_usage",
                {},
            ) or {}
            evidence_meta = data["metadata"]["evidence"]
            return {
                "answer": None,
                "plan": None,
                "rounds": [
                    {
                        "round": 1,
                        "purpose": "initial",
                        "tasks": [
                            {
                                "query_id": "Q1",
                                "query": case["question"],
                                "aspect_ids": [],
                                "success": True,
                                "chunk_ids": [
                                    item["chunk_id"] for item in data["retrieved_chunks"]
                                ],
                                "new_chunk_count": len(data["retrieved_chunks"]),
                                "latency_ms": data["metadata"]["latency_ms"],
                            }
                        ],
                        "candidate_chunk_ids": [
                            item["chunk_id"] for item in data["retrieved_chunks"]
                        ],
                        "candidate_count": len(data["retrieved_chunks"]),
                        "new_chunk_count": len(data["retrieved_chunks"]),
                        "latency_ms": data["metadata"]["latency_ms"],
                    }
                ],
                "evidence_pack": pack_from_data(data),
                "metadata": {
                    "trace_id": data["metadata"].get("trace_id"),
                    "log_id": data["metadata"]["log_id"],
                    "initial_evidence_status": evidence_meta.get("status") or "missing",
                    "stop_reason": "evidence_complete"
                    if evidence_meta.get("status") == "complete"
                    else "max_rounds",
                    "round_count": 1,
                    "retrieval_task_count": 1,
                    "llm_call_count": int(evidence_meta.get("executed", False)),
                    "input_tokens": int(evidence_usage.get("prompt_tokens", 0) or 0),
                    "output_tokens": int(evidence_usage.get("completion_tokens", 0) or 0),
                    "latency_ms": data["metadata"]["latency_ms"]
                    + int(evidence_meta.get("latency_ms", 0) or 0),
                    "degraded": bool(evidence_meta.get("degraded", False)),
                    "errors": [evidence_meta["error"]] if evidence_meta.get("error") else [],
                    "stages": [],
                },
            }

        service = ResearchService(db)
        response = await service.research(
            TENANT_ID,
            ResearchRequest(
                kb_ids=case["kb_ids"],
                user_id="opt18-acceptance",
                query=case["question"],
            ),
            allow_supplement=mode == "research",
        )
        return response.model_dump(mode="json")


async def run_cases(
    *,
    mode: Mode,
    dataset_path: Path,
    output: Path,
    resume: bool,
    maximum: int | None,
    case_types: list[str] | None,
) -> dict[str, Any]:
    dataset = load_json(dataset_path)
    validation = validate_dataset(dataset)
    if not validation["valid"]:
        raise ValueError(f"dataset validation failed: {validation['errors']}")
    expected_meta = run_metadata(dataset_path, mode)
    expected_meta["case_types"] = case_types or list(CASE_TYPES)
    expected_meta["max_cases"] = maximum
    if resume:
        result = load_json(output)
        for field in (
            "dataset_sha256",
            "code_commit",
            "code_sha256",
            "mode",
            "models",
            "prompt_sha256",
            "kb_ids",
            "index_versions",
            "case_types",
            "max_cases",
        ):
            if result.get(field) != expected_meta.get(field):
                raise ValueError(f"resume metadata mismatch: {field}")
    else:
        if output.exists():
            raise FileExistsError(f"output already exists; use --resume: {output}")
        result = {**expected_meta, "cases": []}

    selected = select_cases(dataset["cases"], maximum, case_types)
    if not selected:
        raise ValueError("case selector matched no cases")
    completed = {item["case_id"] for item in result["cases"]}
    for case in selected:
        if case["id"] in completed:
            continue
        started = datetime.now(UTC)
        try:
            data = await asyncio.wait_for(
                execute_case(mode, case), timeout=CASE_TIMEOUT_SECONDS
            )
            metadata = data["metadata"]
            record = {
                "case_id": case["id"],
                "case_type": case["case_type"],
                "question": case["question"],
                "success": True,
                "degraded": bool(metadata.get("degraded", False)),
                "error": None,
                "elapsed_ms": int(
                    (datetime.now(UTC) - started).total_seconds() * 1000
                ),
                **data,
            }
        except TimeoutError:
            record = {
                "case_id": case["id"],
                "case_type": case["case_type"],
                "question": case["question"],
                "success": False,
                "degraded": True,
                "error": "case timeout after 90 seconds",
                "elapsed_ms": CASE_TIMEOUT_SECONDS * 1000,
            }
        except Exception as exc:  # noqa: BLE001 - Persist failure and continue the batch.
            record = {
                "case_id": case["id"],
                "case_type": case["case_type"],
                "question": case["question"],
                "success": False,
                "degraded": True,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "elapsed_ms": int(
                    (datetime.now(UTC) - started).total_seconds() * 1000
                ),
            }
        result["cases"].append(record)
        result["checkpoint_at"] = utc_now()
        save_json(output, result)
        print(
            f"[{mode}] {len(result['cases'])}/{len(selected)} {case['id']}: "
            f"{'ok' if record['success'] else 'failed'}",
            flush=True,
        )
    result["run_finished_at"] = utc_now()
    save_json(output, result)
    return result


def core_chunk_ids(pack: dict[str, Any] | None) -> set[str]:
    if not pack:
        return set()
    item_map = {
        str(item.get("evidence_id")): str(item.get("chunk_id"))
        for item in pack.get("items") or []
    }
    return {
        item_map[str(reference.get("evidence_id"))]
        for group in pack.get("groups") or []
        for reference in group.get("evidence") or []
        if reference.get("role") == "core"
        and str(reference.get("evidence_id")) in item_map
    }


def evidence_chunk_ids(pack: dict[str, Any] | None) -> set[str]:
    return {str(item.get("chunk_id")) for item in (pack or {}).get("items") or []}


def answer_references_valid(answer: str | None, pack: dict[str, Any] | None) -> bool:
    if not answer:
        return True
    import re

    known = {str(item.get("evidence_id")) for item in (pack or {}).get("items") or []}
    return all(reference in known for reference in re.findall(r"\[(E\d+)\]", answer))


def actual_status(covered: int, total: int) -> str:
    return "complete" if covered == total else "missing" if covered == 0 else "partial"


def traceable_ids(ids: set[str], *, case: dict[str, Any]) -> set[str]:
    if not ids:
        return set()
    with SessionLocal() as db:
        kb_id = case["kb_ids"][0]
        index_version = case["index_versions"][kb_id]
        return set(
            db.scalars(
                select(Chunk.id).where(
                    Chunk.id.in_(ids),
                    Chunk.tenant_id == TENANT_ID,
                    Chunk.kb_id.in_(case["kb_ids"]),
                    Chunk.index_version == index_version,
                )
            ).all()
        )


def evaluate_case(case: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    if not record.get("success"):
        return {
            "case_id": case["id"],
            "case_type": case["case_type"],
            "success": False,
            "degraded": True,
            "error": record.get("error"),
            "covered_aspects": 0,
            "aspect_count": len(case["required_aspects"]),
            "complete": False,
            "evidence_precision": 0.0,
            "actual_status": "missing",
            "predicted_status": None,
            "status_match": False,
            "citation_traceability": 1.0,
            "answer_references_valid": True,
            "control_violations": [],
            "latency_ms": record.get("elapsed_ms", 0),
            "round_count": 0,
            "retrieval_task_count": 0,
            "llm_call_count": 0,
            "total_tokens": 0,
        }
    pack = record.get("evidence_pack")
    core_ids = core_chunk_ids(pack)
    item_ids = evidence_chunk_ids(pack)
    covered_flags = [
        bool(core_ids & set(value.get("acceptable_chunk_ids") or []))
        for value in case["required_aspects"]
    ]
    covered = sum(covered_flags)
    valid_ids = {
        str(chunk_id)
        for value in case["required_aspects"]
        for chunk_id in value.get("acceptable_chunk_ids") or []
    }
    status = actual_status(covered, len(covered_flags))
    predicted = (pack or {}).get("status")
    traced = traceable_ids(item_ids, case=case)
    forbidden = set((case.get("control") or {}).get("forbidden_chunk_ids") or [])
    violations = sorted(item_ids & forbidden)
    metadata = record.get("metadata") or {}
    return {
        "case_id": case["id"],
        "case_type": case["case_type"],
        "success": True,
        "degraded": bool(record.get("degraded")),
        "error": record.get("error"),
        "covered_aspects": covered,
        "aspect_count": len(covered_flags),
        "covered_aspect_ids": [
            value["aspect_id"]
            for value, flag in zip(case["required_aspects"], covered_flags, strict=True)
            if flag
        ],
        "complete": status == "complete",
        "evidence_precision": len(item_ids & valid_ids) / len(item_ids) if item_ids else 0.0,
        "actual_status": status,
        "predicted_status": predicted,
        "status_match": status == predicted,
        "evidence_chunk_ids": sorted(item_ids),
        "citation_traceability": len(traced) / len(item_ids) if item_ids else 1.0,
        "answer_references_valid": answer_references_valid(record.get("answer"), pack),
        "control_violations": violations,
        "initial_evidence_status": metadata.get("initial_evidence_status"),
        "stop_reason": metadata.get("stop_reason"),
        "latency_ms": int(metadata.get("latency_ms", record.get("elapsed_ms", 0)) or 0),
        "round_count": int(metadata.get("round_count", 0) or 0),
        "retrieval_task_count": int(metadata.get("retrieval_task_count", 0) or 0),
        "llm_call_count": int(metadata.get("llm_call_count", 0) or 0),
        "total_tokens": int(metadata.get("input_tokens", 0) or 0)
        + int(metadata.get("output_tokens", 0) or 0),
    }


def round_chunk_ids(record: dict[str, Any], round_number: int) -> set[str]:
    result: set[str] = set()
    for round_item in record.get("rounds") or []:
        if round_item.get("round") != round_number:
            continue
        candidate_ids = round_item.get("candidate_chunk_ids")
        if candidate_ids is not None:
            result.update(str(chunk_id) for chunk_id in candidate_ids)
            continue
        result.update(
            str(chunk_id)
            for task in round_item.get("tasks") or []
            for chunk_id in task.get("chunk_ids") or []
        )
    result.update(
        str(item.get("chunk_id"))
        for item in (record.get("evidence_pack") or {}).get("items") or []
        if round_number in (item.get("rounds") or []) and item.get("chunk_id")
    )
    return result


def smoke_check(
    dataset: dict[str, Any],
    research: dict[str, Any],
    decomposition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = validate_dataset(dataset, check_database=False)
    if not validation["valid"] or dataset.get("evaluation_scope") != "smoke":
        raise ValueError(f"invalid smoke dataset: {validation['errors']}")
    if decomposition is not None:
        assert_comparable([decomposition, research])
    expected_ids = {case["id"] for case in dataset["cases"]}
    research_records = {
        item["case_id"]: item for item in research.get("cases") or []
    }
    if set(research_records) != expected_ids:
        raise ValueError("research smoke result does not exactly match the smoke dataset")
    research_rows = {
        case["id"]: evaluate_case(case, research_records[case["id"]])
        for case in dataset["cases"]
    }
    case_checks: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        case_id = case["id"]
        record = research_records[case_id]
        row = research_rows[case_id]
        targets = set((case.get("dependency") or {}).get("target_chunk_ids") or [])
        initial_ids = round_chunk_ids(record, 1)
        supplement_ids = round_chunk_ids(record, 2)
        initial_target_ids = sorted(targets & initial_ids)
        new_target_ids = sorted(targets & (supplement_ids - initial_ids))
        repaired = (
            row["success"]
            and row.get("initial_evidence_status") != "complete"
            and row["round_count"] == 2
            and bool(new_target_ids)
            and row["complete"]
        )
        case_checks.append(
            {
                "case_id": case_id,
                "success": row["success"],
                "degraded": row["degraded"],
                "errors": (record.get("metadata") or {}).get("errors") or [],
                "initial_evidence_status": row.get("initial_evidence_status"),
                "round_count": row["round_count"],
                "llm_call_count": row["llm_call_count"],
                "initial_target_chunk_ids": initial_target_ids,
                "new_target_chunk_ids_in_round_2": new_target_ids,
                "covered_aspects": row["covered_aspects"],
                "aspect_count": row["aspect_count"],
                "actual_status": row["actual_status"],
                "predicted_status": row["predicted_status"],
                "research_repaired": repaired,
            }
        )
    repaired_ids = [item["case_id"] for item in case_checks if item["research_repaired"]]
    initial_target_case_count = sum(
        bool(item["initial_target_chunk_ids"]) for item in case_checks
    )
    supplement_case_count = sum(item["round_count"] == 2 for item in case_checks)
    new_target_case_count = sum(
        bool(item["new_target_chunk_ids_in_round_2"]) for item in case_checks
    )
    hard_limits_ok = all(
        row["success"]
        and not row["degraded"]
        and row["round_count"] <= 2
        and row["llm_call_count"] <= 3
        and row["answer_references_valid"]
        and not row["control_violations"]
        for row in research_rows.values()
    )
    c_gate_passed = len(repaired_ids) >= 2 and hard_limits_ok
    if initial_target_case_count == len(case_checks) and new_target_case_count == 0:
        diagnosis = (
            "四题的目标证据均已由首轮检索、图谱注入或上下文展开带入；"
            "触发第二轮的题也没有新增目标证据，当前数据不能证明Agentic补证带来独立增益。"
        )
        next_action = "按小步门禁停止，不运行B组和完整A/B/C；先扩充真实的检索缺口样本。"
    else:
        diagnosis = "C组未达到至少两题真实补证修复的门槛。"
        next_action = "检查未命中的依赖链和补证查询后，再决定是否运行B组。"
    comparison: dict[str, Any] | None = None
    passed = c_gate_passed
    if decomposition is not None:
        decomposition_records = {
            item["case_id"]: item for item in decomposition.get("cases") or []
        }
        if set(decomposition_records) != expected_ids:
            raise ValueError("decomposition smoke result does not exactly match the smoke dataset")
        decomposition_rows = {
            case["id"]: evaluate_case(case, decomposition_records[case["id"]])
            for case in dataset["cases"]
        }
        denominator = [
            case["id"]
            for case in dataset["cases"]
            if not decomposition_rows[case["id"]]["complete"]
        ]
        repaired_from_b = [case_id for case_id in denominator if case_id in repaired_ids]
        repair_rate = len(repaired_from_b) / len(denominator) if denominator else None
        b_aggregate = aggregate(list(decomposition_rows.values()))
        c_aggregate = aggregate(list(research_rows.values()))
        comparison = {
            "b_incomplete_case_ids": denominator,
            "c_repaired_case_ids": repaired_from_b,
            "multi_hop_repair_rate": repair_rate,
            "b_required_aspect_coverage": b_aggregate["required_aspect_coverage"],
            "c_required_aspect_coverage": c_aggregate["required_aspect_coverage"],
        }
        passed = (
            c_gate_passed
            and len(denominator) >= 2
            and repair_rate is not None
            and repair_rate >= 0.5
            and c_aggregate["required_aspect_coverage"]
            >= b_aggregate["required_aspect_coverage"]
        )
    return {
        "name": "opt18_dependent_reference_smoke_gate",
        "generated_at": utc_now(),
        "dataset_name": dataset["name"],
        "dataset_sha256": research.get("dataset_sha256"),
        "criteria": (
            "至少2/4题首轮不完整、第二轮新增命中目标Chunk且最终完整；"
            "提供B组后还要求B未完整样本不少于2题、C修复率不低于50%，且C覆盖率不低于B。"
        ),
        "c_gate_passed": c_gate_passed,
        "hard_limits_ok": hard_limits_ok,
        "initial_target_case_count": initial_target_case_count,
        "supplement_case_count": supplement_case_count,
        "new_target_case_count": new_target_case_count,
        "diagnosis": diagnosis,
        "next_action": next_action,
        "research_repaired_case_ids": repaired_ids,
        "case_checks": case_checks,
        "comparison": comparison,
        "passed": passed,
    }


def render_smoke_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 优化十八依赖型多跳冒烟报告",
        "",
        f"- 测试时间：{report['generated_at']}",
        f"- 结论：**{'通过' if report['passed'] else '未通过'}**",
        f"- C组门禁：{'通过' if report['c_gate_passed'] else '未通过'}",
        f"- 硬限制：{'通过' if report['hard_limits_ok'] else '未通过'}",
        f"- 判定标准：{report['criteria']}",
        f"- 诊断：{report['diagnosis']}",
        f"- 执行决策：{report['next_action']}",
        "",
        "## C组逐题结果",
        "",
        "| Case | 首轮状态 | 轮数 | 首轮已有目标 | 第二轮新增目标 | 覆盖 | 修复成功 |",
        "|---|---|---:|---|---|---:|---|",
    ]
    for item in report["case_checks"]:
        lines.append(
            f"| {item['case_id']} | {item['initial_evidence_status']} | "
            f"{item['round_count']} | {', '.join(item['initial_target_chunk_ids']) or '无'} | "
            f"{', '.join(item['new_target_chunk_ids_in_round_2']) or '无'} | "
            f"{item['covered_aspects']}/{item['aspect_count']} | "
            f"{'是' if item['research_repaired'] else '否'} |"
        )
    degraded = [item for item in report["case_checks"] if item["degraded"]]
    lines.extend(
        [
            "",
            "## 调用与异常",
            "",
            f"- 触发第二轮：{report['supplement_case_count']}/4",
            f"- 第二轮新增目标证据：{report['new_target_case_count']}/4",
            f"- LLM 调用总数：{sum(item['llm_call_count'] for item in report['case_checks'])}",
            f"- 降级 Case：{', '.join(item['case_id'] for item in degraded) or '无'}",
        ]
    )
    comparison = report.get("comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## B/C 增量",
                "",
                f"- B组未完整：{', '.join(comparison['b_incomplete_case_ids']) or '无'}",
                f"- C组修复：{', '.join(comparison['c_repaired_case_ids']) or '无'}",
                f"- 多跳修复率：{pct(comparison['multi_hop_repair_rate'])}",
                f"- Required Aspect Coverage：B {pct(comparison['b_required_aspect_coverage'])}；C {pct(comparison['c_required_aspect_coverage'])}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aspect_count = sum(row["aspect_count"] for row in rows)
    evidence_weight = sum(len(row.get("evidence_chunk_ids") or []) for row in rows)
    precision_hits = sum(
        row["evidence_precision"] * len(row.get("evidence_chunk_ids") or []) for row in rows
    )
    latencies = [row["latency_ms"] for row in rows if row["success"]]
    return {
        "case_count": len(rows),
        "required_aspect_coverage": (
            sum(row["covered_aspects"] for row in rows) / aspect_count if aspect_count else 0.0
        ),
        "complete_case_rate": sum(row["complete"] for row in rows) / len(rows) if rows else 0.0,
        "complete_case_count": sum(row["complete"] for row in rows),
        "evidence_precision": precision_hits / evidence_weight if evidence_weight else 0.0,
        "status_accuracy": sum(row["status_match"] for row in rows) / len(rows) if rows else 0.0,
        "citation_traceability": min(
            (row["citation_traceability"] for row in rows), default=1.0
        ),
        "p50_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
        "average_round_count": statistics.mean(row["round_count"] for row in rows) if rows else 0,
        "average_retrieval_task_count": (
            statistics.mean(row["retrieval_task_count"] for row in rows) if rows else 0
        ),
        "average_llm_call_count": (
            statistics.mean(row["llm_call_count"] for row in rows) if rows else 0
        ),
        "average_tokens": statistics.mean(row["total_tokens"] for row in rows) if rows else 0,
        "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else 0.0,
        "degraded_rate": sum(row["degraded"] for row in rows) / len(rows) if rows else 0.0,
    }


def assert_comparable(runs: list[dict[str, Any]]) -> None:
    fields = (
        "dataset_sha256",
        "code_commit",
        "code_sha256",
        "models",
        "prompt_sha256",
        "tenant_id",
        "profile",
        "kb_ids",
        "index_versions",
    )
    for field in fields:
        values = [run.get(field) for run in runs]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"A/B/C metadata mismatch: {field}")


def decide_conclusion(checks: dict[str, bool]) -> str:
    query_gain = (
        checks["independent_coverage_b_gt_a"]
        and checks["independent_complete_b_adds_one"]
    )
    repair_gain = (
        checks["dependent_complete_c_gt_b"]
        and checks["dependent_complete_c_adds_one"]
        and checks["multi_hop_repair_evaluable"]
        and checks["multi_hop_repair_at_least_50pct"]
    )
    quality_and_control = all(
        checks[name]
        for name in (
            "simple_regression_within_5pp",
            "c_coverage_not_below_b",
            "c_precision_at_least_80pct",
            "c_precision_drop_within_5pp",
            "c_status_accuracy_at_least_80pct",
            "citation_traceability_100pct",
            "unnecessary_supplement_at_most_10pct",
            "agent_hard_limits",
        )
    )
    if query_gain and repair_gain and quality_and_control:
        return "通过"
    if query_gain and repair_gain and checks["agent_hard_limits"]:
        return "有条件通过"
    # The acceptance document explicitly forbids a conditional pass when
    # decomposition has no gain or the second round repairs no multi-hop case.
    return "不通过"


def validate_langfuse_traces(research: dict[str, Any]) -> dict[str, Any]:
    from app.observability.langfuse_client import get_langfuse_client

    client = get_langfuse_client()
    if client is None:
        return {"status": "not_evaluable", "checks": [], "errors": ["Langfuse不可用"]}
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    cases = research.get("cases") or []
    for expected_rounds in (1, 2):
        record = next(
            (
                item
                for item in cases
                if (item.get("metadata") or {}).get("round_count") == expected_rounds
                and (item.get("metadata") or {}).get("trace_id")
            ),
            None,
        )
        if record is None:
            errors.append(f"没有可核验的{expected_rounds}轮Trace")
            continue
        metadata = record["metadata"]
        try:
            trace = client.fetch_trace(metadata["trace_id"]).data
            span_names = {
                str(getattr(observation, "name", ""))
                for observation in (getattr(trace, "observations", None) or [])
            }
            expected_spans = {
                "planner",
                "retrieval_round_1",
                "evidence_decider",
                "finalizer",
            }
            if expected_rounds == 2:
                expected_spans.add("retrieval_round_2")
            trace_metadata = getattr(trace, "metadata", None) or {}
            trace_output = getattr(trace, "output", None) or {}
            passed = (
                expected_spans.issubset(span_names)
                and trace_metadata.get("tenant_id") == TENANT_ID
                and trace_metadata.get("log_id") == metadata.get("log_id")
                and trace_output.get("round_count") == expected_rounds
                and trace_output.get("stop_reason") == metadata.get("stop_reason")
            )
            checks.append(
                {
                    "case_id": record["case_id"],
                    "round_count": expected_rounds,
                    "trace_id": metadata["trace_id"],
                    "span_names": sorted(span_names),
                    "stop_reason": metadata.get("stop_reason"),
                    "passed": passed,
                }
            )
            if not passed:
                errors.append(f"{record['case_id']}: Trace字段或Span不一致")
        except Exception as exc:  # noqa: BLE001 - observability is optional to core metrics.
            errors.append(f"{record['case_id']}: {type(exc).__name__}: {str(exc)[:200]}")
    return {"status": "passed" if checks and not errors else "failed", "checks": checks, "errors": errors}


def compare_runs(
    dataset: dict[str, Any],
    single: dict[str, Any],
    decomposition: dict[str, Any],
    research: dict[str, Any],
) -> dict[str, Any]:
    runs = {"single": single, "decomposition": decomposition, "research": research}
    assert_comparable(list(runs.values()))
    expected_ids = {case["id"] for case in dataset["cases"]}
    evaluated: dict[str, list[dict[str, Any]]] = {}
    by_mode_id: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, run in runs.items():
        records = {item["case_id"]: item for item in run.get("cases") or []}
        if set(records) != expected_ids:
            missing = sorted(expected_ids - set(records))
            extra = sorted(set(records) - expected_ids)
            raise ValueError(f"{mode} cases are incomplete; missing={missing}, extra={extra}")
        rows = [evaluate_case(case, records[case["id"]]) for case in dataset["cases"]]
        evaluated[mode] = rows
        by_mode_id[mode] = {row["case_id"]: row for row in rows}

    summary = {mode: aggregate(rows) for mode, rows in evaluated.items()}
    layered = {
        case_type: {
            mode: aggregate([row for row in rows if row["case_type"] == case_type])
            for mode, rows in evaluated.items()
        }
        for case_type in CASE_TYPES
    }
    dependent = [case for case in dataset["cases"] if case["case_type"] == "dependent_multi_hop"]
    repair_denominator = [
        case
        for case in dependent
        if not by_mode_id["decomposition"][case["id"]]["complete"]
        and (case.get("dependency") or {}).get("target_chunk_ids")
    ]
    repaired = [
        case
        for case in repair_denominator
        if by_mode_id["research"][case["id"]]["complete"]
        and by_mode_id["research"][case["id"]]["round_count"] == 2
    ]
    initial_complete = [
        row for row in evaluated["research"] if row.get("initial_evidence_status") == "complete"
    ]
    unnecessary = [row for row in initial_complete if row["round_count"] > 1]
    agentic = {
        "multi_hop_repair_rate": (
            len(repaired) / len(repair_denominator) if repair_denominator else None
        ),
        "multi_hop_repair_denominator": len(repair_denominator),
        "multi_hop_repaired_cases": [case["id"] for case in repaired],
        "multi_hop_unrepaired_cases": [
            case["id"] for case in repair_denominator if case not in repaired
        ],
        "unnecessary_supplement_rate": (
            len(unnecessary) / len(initial_complete) if initial_complete else None
        ),
        "initial_complete_case_count": len(initial_complete),
        "unnecessary_supplement_cases": [row["case_id"] for row in unnecessary],
    }
    independent = layered["independent_multi_aspect"]
    simple = layered["simple"]
    dependent_layer = layered["dependent_multi_hop"]
    c_summary = summary["research"]
    b_summary = summary["decomposition"]
    hard_limit_violations = [
        f"{row['case_id']}: control or hard limit"
        for row in evaluated["research"]
        if row["round_count"] > 2
        or row["llm_call_count"] > 3
        or row["latency_ms"] > CASE_TIMEOUT_SECONDS * 1000
        or row["control_violations"]
        or not row["answer_references_valid"]
    ]
    checks = {
        "independent_coverage_b_gt_a": independent["decomposition"][
            "required_aspect_coverage"
        ]
        > independent["single"]["required_aspect_coverage"],
        "independent_complete_b_adds_one": independent["decomposition"][
            "complete_case_count"
        ]
        >= independent["single"]["complete_case_count"] + 1,
        "simple_regression_within_5pp": min(
            simple["decomposition"]["complete_case_rate"],
            simple["research"]["complete_case_rate"],
        )
        >= simple["single"]["complete_case_rate"] - 0.05,
        "dependent_complete_c_gt_b": dependent_layer["research"]["complete_case_rate"]
        > dependent_layer["decomposition"]["complete_case_rate"],
        "dependent_complete_c_adds_one": dependent_layer["research"][
            "complete_case_count"
        ]
        >= dependent_layer["decomposition"]["complete_case_count"] + 1,
        "multi_hop_repair_evaluable": len(repair_denominator) >= 2,
        "multi_hop_repair_at_least_50pct": agentic["multi_hop_repair_rate"] is not None
        and agentic["multi_hop_repair_rate"] >= 0.5,
        "c_coverage_not_below_b": c_summary["required_aspect_coverage"]
        >= b_summary["required_aspect_coverage"],
        "c_precision_at_least_80pct": c_summary["evidence_precision"] >= 0.8,
        "c_precision_drop_within_5pp": c_summary["evidence_precision"]
        >= b_summary["evidence_precision"] - 0.05,
        "c_status_accuracy_at_least_80pct": c_summary["status_accuracy"] >= 0.8,
        "citation_traceability_100pct": all(
            value["citation_traceability"] == 1.0 for value in summary.values()
        ),
        "unnecessary_supplement_at_most_10pct": agentic[
            "unnecessary_supplement_rate"
        ]
        is not None
        and agentic["unnecessary_supplement_rate"] <= 0.1,
        "agent_hard_limits": not hard_limit_violations,
    }
    conclusion = decide_conclusion(checks)
    observability = validate_langfuse_traces(research)
    return {
        "name": "opt18_agentic_research_comparison",
        "generated_at": utc_now(),
        "dataset": {
            "name": dataset["name"],
            "sha256": single["dataset_sha256"],
            "case_count": len(dataset["cases"]),
            "type_counts": dict(Counter(case["case_type"] for case in dataset["cases"])),
            "annotation": dataset["annotation"],
        },
        "runtime": {
            "code_commit": single["code_commit"],
            "code_dirty": single["code_dirty"],
            "code_sha256": single["code_sha256"],
            "models": single["models"],
            "prompt_sha256": single["prompt_sha256"],
            "index_versions": single["index_versions"],
            "cost": None,
            "cost_note": "未冻结可核验的模型价格版本，仅报告真实Token，不估算金额。",
        },
        "definitions": {
            "A": "Single Retrieve：原问题单次检索和证据编排",
            "B": "Decomposition：Planner并行首轮检索，禁止补证",
            "C": "Research：Planner并行检索，缺失时最多一次补证",
        },
        "summary": summary,
        "layered": layered,
        "agentic": agentic,
        "checks": checks,
        "hard_limit_violations": hard_limit_violations,
        "failed_or_degraded": {
            mode: [
                row["case_id"]
                for row in rows
                if not row["success"] or row["degraded"]
            ]
            for mode, rows in evaluated.items()
        },
        "observability": observability,
        "cases": evaluated,
        "conclusion": conclusion,
    }


def pct(value: float | None) -> str:
    return "not_evaluable" if value is None else f"{value * 100:.2f}%"


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def render_markdown(report: dict[str, Any]) -> str:
    a, b, c = (report["summary"][key] for key in ("single", "decomposition", "research"))
    agentic = report["agentic"]
    lines = [
        "# 优化十八受控 Agentic Research A/B/C 效果报告",
        "",
        f"- 测试时间：{report['generated_at']}",
        f"- 结论：**{report['conclusion']}**",
        f"- Case：{report['dataset']['case_count']}，分布 {report['dataset']['type_counts']}",
        f"- 数据集 SHA-256：`{report['dataset']['sha256']}`",
        f"- 代码 commit：`{report['runtime']['code_commit']}`",
        f"- 代码指纹：`{report['runtime']['code_sha256']}`",
        f"- 模型：FAST `{report['runtime']['models']['fast']}`；STRONG `{report['runtime']['models']['strong']}`",
        f"- 索引版本：{report['runtime']['index_versions']}",
        "",
        "## 主效果表",
        "",
        "| 指标 | A Single | B Decomposition | C Research | 主要对比 |",
        "|---|---:|---:|---:|---|",
        f"| Required Aspect Coverage | {pct(a['required_aspect_coverage'])} | {pct(b['required_aspect_coverage'])} | {pct(c['required_aspect_coverage'])} | B-A、C-B |",
        f"| Complete Case Rate | {pct(a['complete_case_rate'])} | {pct(b['complete_case_rate'])} | {pct(c['complete_case_rate'])} | B-A、C-B |",
        f"| Evidence Precision | {pct(a['evidence_precision'])} | {pct(b['evidence_precision'])} | {pct(c['evidence_precision'])} | 质量护栏 |",
        f"| Status Accuracy | {pct(a['status_accuracy'])} | {pct(b['status_accuracy'])} | {pct(c['status_accuracy'])} | 决策可靠性 |",
        f"| Citation Traceability | {pct(a['citation_traceability'])} | {pct(b['citation_traceability'])} | {pct(c['citation_traceability'])} | 硬性门槛 |",
        f"| Multi-hop Repair Rate | — | — | {pct(agentic['multi_hop_repair_rate'])} | C修复能力 |",
        f"| Unnecessary Supplement Rate | — | — | {pct(agentic['unnecessary_supplement_rate'])} | C控制能力 |",
        "",
        "## 性能与成本",
        "",
        "| 观察项 | A Single | B Decomposition | C Research |",
        "|---|---:|---:|---:|",
        f"| P50 / P95 延迟（ms） | {number(a['p50_latency_ms'])} / {number(a['p95_latency_ms'])} | {number(b['p50_latency_ms'])} / {number(b['p95_latency_ms'])} | {number(c['p50_latency_ms'])} / {number(c['p95_latency_ms'])} |",
        f"| 平均检索轮数 | {number(a['average_round_count'])} | {number(b['average_round_count'])} | {number(c['average_round_count'])} |",
        f"| 平均检索任务数 | {number(a['average_retrieval_task_count'])} | {number(b['average_retrieval_task_count'])} | {number(c['average_retrieval_task_count'])} |",
        f"| 平均 LLM 调用数 | {number(a['average_llm_call_count'])} | {number(b['average_llm_call_count'])} | {number(c['average_llm_call_count'])} |",
        f"| 平均 Token | {number(a['average_tokens'])} | {number(b['average_tokens'])} | {number(c['average_tokens'])} |",
        f"| 成功率 / 降级率 | {pct(a['success_rate'])} / {pct(a['degraded_rate'])} | {pct(b['success_rate'])} / {pct(b['degraded_rate'])} | {pct(c['success_rate'])} / {pct(c['degraded_rate'])} |",
        "",
        f"- 成本：{report['runtime']['cost_note']}",
        "",
        "## 分层结果",
        "",
        "| Case 类型 | A Coverage / Complete | B Coverage / Complete | C Coverage / Complete |",
        "|---|---:|---:|---:|",
    ]
    for case_type in CASE_TYPES:
        layer = report["layered"][case_type]
        lines.append(
            f"| {case_type} | {pct(layer['single']['required_aspect_coverage'])} / {pct(layer['single']['complete_case_rate'])} | "
            f"{pct(layer['decomposition']['required_aspect_coverage'])} / {pct(layer['decomposition']['complete_case_rate'])} | "
            f"{pct(layer['research']['required_aspect_coverage'])} / {pct(layer['research']['complete_case_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 多跳与控制明细",
            "",
            f"- B组未完整且可补证：{agentic['multi_hop_repair_denominator']} 题",
            f"- C组修复：{', '.join(agentic['multi_hop_repaired_cases']) or '无'}",
            f"- 未修复：{', '.join(agentic['multi_hop_unrepaired_cases']) or '无'}",
            f"- 首轮已完整：{agentic['initial_complete_case_count']} 题",
            f"- 不必要补证：{', '.join(agentic['unnecessary_supplement_cases']) or '无'}",
            "",
            "## 验收门槛",
            "",
        ]
    )
    lines.extend(
        f"- {'通过' if passed else '未通过'}：{name}"
        for name, passed in report["checks"].items()
    )
    lines.extend(
        [
            "",
            "## 失败、降级与限制",
            "",
            f"- A：{report['failed_or_degraded']['single'] or '无'}",
            f"- B：{report['failed_or_degraded']['decomposition'] or '无'}",
            f"- C：{report['failed_or_degraded']['research'] or '无'}",
            f"- 硬上限或权限违规：{report['hard_limit_violations'] or '无'}",
            (
                f"- Langfuse链路验收：{report['observability']['status']}；"
                f"{report['observability']['errors'] or '单轮与双轮Trace结构、stop_reason均一致'}"
            ),
            f"- 标注方法：{report['dataset']['annotation']['method']}",
            "- A不生成最终回答，因此只参与证据、状态、回溯、延迟和调用成本对比。",
            "- 本报告由原始JSON自动计算，未使用RAGAS或语义Judge替代证据标注。",
            "",
            "## 逐题索引",
            "",
        ]
    )
    for mode in ("single", "decomposition", "research"):
        for row in report["cases"][mode]:
            lines.append(
                f"- {mode} / {row['case_id']}：coverage "
                f"{row['covered_aspects']}/{row['aspect_count']}，"
                f"status {row['predicted_status']}，round {row['round_count']}，"
                f"LLM {row['llm_call_count']}，latency {row['latency_ms']} ms"
            )
    lines.append("")
    return "\n".join(lines)


def preflight(dataset_path: Path) -> dict[str, Any]:
    dataset = load_json(dataset_path)
    validation = validate_dataset(dataset)
    case_count = len(dataset.get("cases") or [])
    index_versions = {
        kb_id: version
        for case in dataset.get("cases") or []
        for kb_id, version in case.get("index_versions", {}).items()
    }
    outputs = {
        name: str(PROJECT_ROOT / "eval/opt18_research" / name)
        for name in (
            "single_baseline.json",
            "decomposition_parallel.json",
            "agentic_research.json",
        )
    }
    dependencies = {
        "dataset_valid": validation["valid"],
        "database": not any("database" in item for item in validation["errors"]),
        "elasticsearch": endpoint_available(settings.elasticsearch_url),
        "embedding_api_key": not settings.model_api_key.startswith("your-"),
        "llm_api_key": not settings.llm_api_key.startswith("your-"),
        "rerank_configured": bool(
            settings.rerank_base_url
            and (settings.rerank_api_key or settings.model_api_key)
        ),
        "evidence_enabled": settings.evidence_enabled,
    }
    return {
        "case_count": case_count,
        "type_counts": validation["type_counts"],
        "requests": {"A": case_count, "B": case_count, "C": case_count, "total": case_count * 3},
        "estimated_llm_calls": {
            "A": case_count,
            "B": case_count * 3,
            "C": case_count * 3,
            "upper_bound": case_count * 7,
        },
        "models": {"fast": settings.llm_model, "strong": settings.llm_strong_model},
        "index_versions": index_versions,
        "dependencies": dependencies,
        "validation_errors": validation["errors"],
        "outputs_exist": {name: Path(path).exists() for name, path in outputs.items()},
        "note": "preflight未调用LLM；理论上限按每题A一次、B/C各三次计算。",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimization 18 serial A/B/C evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare_smoke = subparsers.add_parser("prepare-smoke")
    prepare_smoke.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--skip-database", action="store_true", help=argparse.SUPPRESS)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--dataset", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=("single", "decomposition", "research"), required=True)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--case-concurrency", type=int, default=1)
    run.add_argument("--max-cases", type=int)
    run.add_argument("--case-type", choices=CASE_TYPES, action="append")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--dataset", type=Path, required=True)
    compare.add_argument("--single", type=Path, required=True)
    compare.add_argument("--decomposition", type=Path, required=True)
    compare.add_argument("--research", type=Path, required=True)
    compare.add_argument("--json-output", type=Path, required=True)
    compare.add_argument("--markdown-output", type=Path, required=True)
    smoke = subparsers.add_parser("smoke-check")
    smoke.add_argument("--dataset", type=Path, required=True)
    smoke.add_argument("--research", type=Path, required=True)
    smoke.add_argument("--decomposition", type=Path)
    smoke.add_argument("--json-output", type=Path, required=True)
    smoke.add_argument("--markdown-output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            save_json(args.output, prepare_dataset())
            print(f"saved: {args.output}")
        elif args.command == "prepare-smoke":
            save_json(args.output, prepare_smoke_dataset())
            print(f"saved: {args.output}")
        elif args.command == "validate":
            result = validate_dataset(
                load_json(args.dataset), check_database=not args.skip_database
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["valid"] else 1
        elif args.command == "preflight":
            result = preflight(args.dataset)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if all(result["dependencies"].values()) else 1
        elif args.command == "run":
            if args.repeat != 1 or args.warmup != 0 or args.case_concurrency != 1:
                raise ValueError("formal evaluator requires repeat=1, warmup=0, case-concurrency=1")
            with OutputLock(args.output):
                asyncio.run(
                    run_cases(
                        mode=args.mode,
                        dataset_path=args.dataset,
                        output=args.output,
                        resume=args.resume,
                        maximum=args.max_cases,
                        case_types=args.case_type,
                    )
                )
        elif args.command == "compare":
            report = compare_runs(
                load_json(args.dataset),
                load_json(args.single),
                load_json(args.decomposition),
                load_json(args.research),
            )
            save_json(args.json_output, report)
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
            print(f"saved: {args.json_output}")
            print(f"saved: {args.markdown_output}")
        elif args.command == "smoke-check":
            report = smoke_check(
                load_json(args.dataset),
                load_json(args.research),
                load_json(args.decomposition) if args.decomposition else None,
            )
            save_json(args.json_output, report)
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(
                render_smoke_markdown(report), encoding="utf-8"
            )
            print(f"saved: {args.json_output}")
            print(f"saved: {args.markdown_output}")
            return 0 if report["passed"] else 1
        return 0
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI returns a concise actionable failure.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
