#!/usr/bin/env python3
"""
Mini-Hermes Agent Runtime - 主入口
"""
import os
import sys
import argparse
from dotenv import load_dotenv

from agent import RAGAgent, build_index, VectorRetriever
from agent.eval import RAGEvaluator, create_sample_test_cases


def cmd_index(args):
    """构建索引子命令"""
    print("[INDEX] 正在构建知识库索引...")
    result = build_index(args.data_dir, args.index_dir)
    print(result)


def cmd_query(args):
    """单次查询子命令"""
    api_key = _get_api_key()
    if not api_key:
        return

    agent = _create_agent(api_key, args)
    print(f"[Q] 问题：{args.question}\n")
    result = agent.run(args.question)
    print(f"\n[A] 回答：\n{result}")


def cmd_interactive(args):
    """交互模式子命令"""
    api_key = _get_api_key()
    if not api_key:
        return

    agent = _create_agent(api_key, args)
    print("=" * 50)
    print("Mini-Hermes Agent Runtime - 交互模式")
    print(f"知识库: {args.data_dir}/")
    print("输入 'exit' 退出, 'stats' 查看统计, 'trace' 查看轨迹")
    print("=" * 50)

    while True:
        try:
            question = input("\n你: ").strip()
            if question.lower() in ("exit", "quit", "退出"):
                print("再见!")
                break
            if question.lower() == "stats":
                from agent.tools import tool_get_stats
                print(f"\n{tool_get_stats()}\n")
                continue
            if question.lower() == "trace":
                trace = agent.get_trace()
                if trace:
                    print(f"\n[TRACE] {trace['total_steps']} 步, "
                          f"耗时 {trace['total_duration_ms']}ms")
                    for s in trace['steps']:
                        print(f"  Step{s['step']}: {s['action']} ({s['duration_ms']}ms)")
                else:
                    print("\n[TRACE] 无轨迹数据")
                continue
            if not question:
                continue

            print("\n思考中...")
            result = agent.chat(question)
            print(f"\nAgent: {result}\n")
        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


def _get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        print("[ERROR] 未设置 API Key")
        print("请设置环境变量 OPENAI_API_KEY 或在 .env 文件中配置")
    return api_key


def _create_agent(api_key: str, args):
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = args.model or os.getenv("MODEL_NAME", "gpt-4o-mini")

    return RAGAgent(
        api_key=api_key,
        base_url=base_url,
        model=model,
        memory_dir=args.memory_dir,
        skills_dir=args.skills_dir,
        data_dir=args.data_dir,
        max_steps=args.max_steps,
    )


def _setup_encoding():
    """Windows 下强制输出 UTF-8"""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def cmd_eval(args):
    """检索质量评估子命令"""
    test_path = args.test_file or os.path.join(args.data_dir, "test_cases.json")

    if not os.path.exists(test_path):
        print(f"[ERROR] 测试用例文件不存在: {test_path}")
        print(f"运行 python main.py gen-test 生成示例测试用例")
        return

    retriever = VectorRetriever(persist_dir=args.index_dir)
    if not retriever.available:
        print("[ERROR] ChromaDB 不可用，请先安装 chromadb")
        return

    evaluator = RAGEvaluator(retriever)
    evaluator.load_test_cases(test_path)
    metrics = evaluator.evaluate(top_k=args.top_k)
    print(evaluator.report(metrics))


def cmd_gen_test(args):
    """生成示例测试用例"""
    result = create_sample_test_cases(args.data_dir)
    print(result)


def main():
    _setup_encoding()
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Mini-Hermes Agent Runtime - 轻量级 RAG 知识库问答智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py index                          # 构建知识库索引
  python main.py query "什么是RAG"              # 单次查询
  python main.py -i                             # 交互模式
  python main.py eval --top-k 5                 # 检索质量评估
  python main.py gen-test                       # 生成测试用例模板
        """
    )

    parser.add_argument("--data-dir", default="data", help="文档目录")
    parser.add_argument("--index-dir", default="chroma_db", help="索引存储目录")
    parser.add_argument("--memory-dir", default="memory", help="记忆文件目录")
    parser.add_argument("--skills-dir", default="skills", help="技能文件目录")
    parser.add_argument("--model", default=None, help="使用的模型")
    parser.add_argument("--max-steps", type=int, default=10, help="最大执行步数")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    parser_index = subparsers.add_parser("index", help="构建知识库索引")
    parser_index.set_defaults(func=cmd_index)

    parser_query = subparsers.add_parser("query", help="单次查询")
    parser_query.add_argument("question", help="要查询的问题")
    parser_query.set_defaults(func=cmd_query)

    parser_interactive = subparsers.add_parser("interactive", aliases=["i"], help="交互模式")
    parser_interactive.set_defaults(func=cmd_interactive)

    parser_eval = subparsers.add_parser("eval", help="检索质量评估 (Precision/Recall/MRR)")
    parser_eval.add_argument("--test-file", default=None, help="测试用例 JSON 文件路径")
    parser_eval.add_argument("--top-k", type=int, default=5, help="评估的 top-k 值 (默认 5)")
    parser_eval.set_defaults(func=cmd_eval)

    parser_gen = subparsers.add_parser("gen-test", help="生成示例测试用例模板")
    parser_gen.set_defaults(func=cmd_gen_test)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
