"""
RAG 检索评估模块
支持精确率、召回率、MRR、NDCG 等指标的计算
"""
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .retriever import VectorRetriever


@dataclass
class TestCase:
    """单个评测用例"""
    query: str
    relevant_sources: List[str]  # 期望命中的文档文件名列表


@dataclass
class QueryResult:
    """单条查询的评测结果"""
    query: str
    retrieved_sources: List[str]      # 实际检索到的来源（按排名）
    relevant_sources: List[str]       # 期望命中的来源
    precision_at_k: Dict[int, float]  # {1: 1.0, 3: 0.67, 5: 0.6}
    recall_at_k: Dict[int, float]     # {1: 0.33, 3: 0.67, 5: 1.0}
    reciprocal_rank: float            # 第一个相关结果的倒数排名
    hit: bool                         # top-k 中是否至少命中一个


class RAGEvaluator:
    """RAG 检索质量评估器

    用法:
        evaluator = RAGEvaluator(retriever)
        evaluator.load_test_cases("data/test_cases.json")
        report = evaluator.evaluate(top_k=5)
        print(evaluator.report(report))
    """

    def __init__(self, retriever: VectorRetriever):
        self.retriever = retriever
        self.test_cases: List[TestCase] = []

    def load_test_cases(self, path: str) -> None:
        """从 JSON 文件加载测试用例

        格式:
        [
            {
                "query": "什么是RAG",
                "relevant_sources": ["RAG技术入门指南.md"]
            },
            ...
        ]
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            self.test_cases.append(TestCase(
                query=item["query"],
                relevant_sources=item["relevant_sources"],
            ))

    def add_test_case(self, query: str, relevant_sources: List[str]) -> None:
        """手动添加单个测试用例"""
        self.test_cases.append(TestCase(
            query=query,
            relevant_sources=relevant_sources,
        ))

    def evaluate(self, top_k: int = 5) -> Dict:
        """对所有测试用例运行评估，返回汇总报告"""
        if not self.test_cases:
            return {"error": "没有测试用例"}

        results: List[QueryResult] = []
        for tc in self.test_cases:
            qr = self._evaluate_one(tc, top_k)
            results.append(qr)

        return self._aggregate(results, top_k)

    def _evaluate_one(self, tc: TestCase, max_k: int) -> QueryResult:
        """评估单条查询"""
        items = self.retriever.search(tc.query, top_k=max_k)
        retrieved_sources = [source for (_, source, _) in items]

        # 计算各 k 值的 Precision 和 Recall
        precision_at_k: Dict[int, float] = {}
        recall_at_k: Dict[int, float] = {}
        reciprocal_rank = 0.0
        hit = False

        rel_set = set(tc.relevant_sources)

        for k in range(1, max_k + 1):
            top_k_sources = retrieved_sources[:k]
            top_k_set = set(top_k_sources)

            # Precision@k = 前 k 个结果中相关文档数 / k
            relevant_in_top_k = len(top_k_set & rel_set)
            precision_at_k[k] = relevant_in_top_k / k

            # Recall@k = 前 k 个结果中相关文档数 / 总相关文档数
            recall_at_k[k] = relevant_in_top_k / len(rel_set) if rel_set else 0

        # MRR: 第一个相关结果的排名倒数
        for rank, src in enumerate(retrieved_sources, 1):
            if src in rel_set:
                reciprocal_rank = 1.0 / rank
                hit = True
                break

        return QueryResult(
            query=tc.query,
            retrieved_sources=retrieved_sources,
            relevant_sources=tc.relevant_sources,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            reciprocal_rank=reciprocal_rank,
            hit=hit,
        )

    def _aggregate(self, results: List[QueryResult], max_k: int) -> Dict:
        """汇总所有查询结果"""
        n = len(results)

        avg_precision_at_k = {}
        avg_recall_at_k = {}
        for k in range(1, max_k + 1):
            avg_precision_at_k[k] = sum(r.precision_at_k[k] for r in results) / n
            avg_recall_at_k[k] = sum(r.recall_at_k[k] for r in results) / n

        mrr = sum(r.reciprocal_rank for r in results) / n
        hit_rate = sum(1 for r in results if r.hit) / n

        # NDCG@k
        ndcg_at_k = {}
        for k in range(1, max_k + 1):
            ndcg_at_k[k] = self._compute_avg_ndcg(results, k)

        return {
            "total_queries": n,
            "top_k": max_k,
            "avg_precision_at_k": avg_precision_at_k,
            "avg_recall_at_k": avg_recall_at_k,
            "ndcg_at_k": ndcg_at_k,
            "mrr": round(mrr, 4),
            "hit_rate": round(hit_rate, 4),
            "details": [
                {
                    "query": r.query,
                    "retrieved": r.retrieved_sources,
                    "expected": r.relevant_sources,
                    "precision@1": r.precision_at_k.get(1, 0),
                    "precision@3": r.precision_at_k.get(3, 0),
                    "precision@5": r.precision_at_k.get(5, 0),
                    "recall@5": r.recall_at_k.get(5, 0),
                    "reciprocal_rank": round(r.reciprocal_rank, 4),
                    "hit": r.hit,
                }
                for r in results
            ],
        }

    def _compute_avg_ndcg(self, results: List[QueryResult], k: int) -> float:
        """计算 NDCG@k"""
        total = 0.0
        for r in results:
            rel_set = set(r.relevant_sources)
            # DCG = sum(rel_i / log2(i+1))
            dcg = 0.0
            for i, src in enumerate(r.retrieved_sources[:k], 1):
                rel = 1.0 if src in rel_set else 0.0
                import math
                dcg += rel / math.log2(i + 1)

            # IDCG = 理想排序下的 DCG
            ideal_rels = sorted(
                [1.0 if s in rel_set else 0.0 for s in r.retrieved_sources[:k]],
                reverse=True
            )
            idcg = 0.0
            for i, rel in enumerate(ideal_rels, 1):
                idcg += rel / math.log2(i + 1)

            ndcg = dcg / idcg if idcg > 0 else 0.0
            total += ndcg

        return round(total / len(results), 4)

    def report(self, metrics: Dict) -> str:
        """生成可读的评估报告"""
        if "error" in metrics:
            return f"[ERROR] {metrics['error']}"

        k = metrics["top_k"]
        lines = [
            "=" * 55,
            f"  RAG 检索评估报告 ({metrics['total_queries']} 条查询, Top-{k})",
            "=" * 55,
            "",
        ]

        for i in [1, 3, 5]:
            if i > k:
                break
            p = metrics["avg_precision_at_k"][i]
            r = metrics["avg_recall_at_k"][i]
            n = metrics["ndcg_at_k"][i]
            lines.append(f"  Precision@{i}: {p:.2%}    Recall@{i}: {r:.2%}    NDCG@{i}: {n:.4f}")

        lines += [
            "",
            f"  MRR:      {metrics['mrr']:.4f}",
            f"  Hit Rate: {metrics['hit_rate']:.2%}",
            "",
            "-" * 55,
            "  逐条详情",
            "-" * 55,
        ]

        for d in metrics["details"]:
            status = "HIT" if d["hit"] else "MISS"
            lines.append(f"  [{status}] {d['query']}")
            lines.append(f"    期望: {d['expected']}")
            lines.append(f"    检索: {d['retrieved'][:3]}{'...' if len(d['retrieved']) > 3 else ''}")
            lines.append(f"    P@1={d['precision@1']:.2%} P@3={d['precision@3']:.2%} RR={d['reciprocal_rank']:.4f}")
            lines.append("")

        return "\n".join(lines)


def create_sample_test_cases(data_dir: str = "data") -> str:
    """生成示例测试用例文件"""
    from .document import load_documents

    docs = load_documents(data_dir)
    if not docs:
        return "没有文档，无法生成测试用例"

    cases = []
    for doc in docs:
        if doc["chunks"]:
            # 用每个文档的第一个文本块中的关键词生成查询
            first_chunk = doc["chunks"][0][:100]
            cases.append({
                "query": f"测试查询 - {doc['filename']}",
                "relevant_sources": [doc["filename"]],
                "note": "请将 query 替换为真实的用户问题",
            })

    test_path = os.path.join(data_dir, "test_cases.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    return f"已生成 {len(cases)} 条测试用例模板 → {test_path}"
