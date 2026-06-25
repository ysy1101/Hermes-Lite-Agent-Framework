"""
Workflow 编排引擎模块
支持多步骤工作流的顺序执行、条件分支与错误恢复
"""
from typing import Callable, Dict, List, Optional

from .registry import ToolRegistry
from .state import AgentState, AgentStatus
from .trace import Trace


class WorkflowStep:
    """工作流中的单个步骤"""

    def __init__(
        self,
        name: str,
        tool_name: str,
        args: dict = None,
        condition: Optional[Callable[[dict], bool]] = None,
        on_error: str = "stop",
    ):
        """
        Args:
            name: 步骤名称
            tool_name: 要调用的工具名称
            args: 工具参数
            condition: 条件函数，接收 {"last_output": ..., "results": [...]} 上下文，
                       返回 True 才执行此步骤
            on_error: 错误处理策略 — "stop" 停止 / "skip" 跳过 / "continue" 继续
        """
        self.name = name
        self.tool_name = tool_name
        self.args = args or {}
        self.condition = condition
        self.on_error = on_error


class Workflow:
    """工作流定义"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[WorkflowStep] = []

    def add_step(self, step: WorkflowStep) -> "Workflow":
        """链式添加步骤"""
        self.steps.append(step)
        return self


class WorkflowEngine:
    """工作流执行引擎"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(
        self,
        workflow: Workflow,
        state: AgentState,
        trace: Optional[Trace] = None,
    ) -> dict:
        """
        执行工作流。

        返回:
            {
                "success": bool,
                "results": [{"step": str, "output": str, "error": str|None, "skipped": bool}],
                "final": str,
            }
        """
        state.status = AgentStatus.RUNNING
        results = []
        last_output = ""

        for i, step in enumerate(workflow.steps):
            state.step = i + 1

            # 条件检查
            if step.condition is not None:
                context = {"last_output": last_output, "results": results}
                if not step.condition(context):
                    results.append({
                        "step": step.name,
                        "output": "",
                        "error": None,
                        "skipped": True,
                    })
                    continue

            # 执行工具
            state.status = AgentStatus.CALLING_TOOL
            try:
                output = self.registry.execute(step.tool_name, step.args)
                state.record_tool_call(step.tool_name, step.args, output)
                last_output = output
                results.append({
                    "step": step.name,
                    "output": output,
                    "error": None,
                })

                if trace:
                    trace.record_step(
                        step=i + 1,
                        thought=f"执行工作流步骤: {step.name}",
                        action=step.tool_name,
                        action_args=step.args,
                        observation=output,
                        duration_ms=0,
                    )

            except Exception as e:
                error_msg = f"执行 {step.name} 时出错：{str(e)}"
                results.append({
                    "step": step.name,
                    "output": "",
                    "error": error_msg,
                })

                if trace:
                    trace.record_step(
                        step=i + 1,
                        thought=f"执行工作流步骤: {step.name} (出错)",
                        action=step.tool_name,
                        action_args=step.args,
                        observation=error_msg,
                        duration_ms=0,
                    )

                if step.on_error == "stop":
                    state.status = AgentStatus.ERROR
                    return {"success": False, "results": results, "final": error_msg}
                elif step.on_error == "skip":
                    continue

        state.status = AgentStatus.FINISHED
        final = results[-1]["output"] if results else "工作流完成"
        return {"success": True, "results": results, "final": final}


def create_knowledge_inspection_workflow(data_dir: str = "data") -> Workflow:
    """创建示例工作流：知识库巡检

    1. 列出所有文档
    2. 获取统计信息
    3. 如果文档数 > 0，执行一次测试搜索
    """
    wf = Workflow(
        name="knowledge_inspection",
        description="知识库巡检工作流：检查文档列表、统计信息，并验证检索功能"
    )

    wf.add_step(WorkflowStep(
        name="list_docs",
        tool_name="list_documents",
        args={"data_dir": data_dir},
        on_error="skip",
    ))

    wf.add_step(WorkflowStep(
        name="get_stats",
        tool_name="get_stats",
        args={},
        on_error="skip",
    ))

    def has_documents(context: dict) -> bool:
        last = context.get("last_output", "")
        return "知识库为空" not in last and last != ""

    wf.add_step(WorkflowStep(
        name="test_search",
        tool_name="search_knowledge",
        args={"query": "测试", "top_k": 3},
        condition=has_documents,
        on_error="skip",
    ))

    return wf
