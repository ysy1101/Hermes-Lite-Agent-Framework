"""
Agent 主循环模块 — ReAct 推理引擎
基于 Thought → Action → Observation 循环，集成状态管理、工具注册与 Trace 记录
"""
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI

from .memory import MemoryManager
from .skills import SkillManager
from .tools import registry
from .state import AgentState, AgentStatus
from .trace import Trace


class RAGAgent:
    """Mini-Hermes Agent Runtime — 轻量级 ReAct Agent"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        memory_dir: str = "memory",
        skills_dir: str = "skills",
        data_dir: str = "data",
        max_steps: int = 10,
        enable_trace: bool = True,
        enable_state: bool = True,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.memory = MemoryManager(memory_dir)
        self.skills = SkillManager(skills_dir)
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.messages: List[Dict[str, Any]] = []

        # 新增组件
        self.enable_trace = enable_trace
        self.enable_state = enable_state
        self.registry = registry
        self.state = AgentState(max_steps=max_steps)
        self.trace = Trace(enabled=enable_trace)

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        context = self.memory.get_system_context()

        prompt = f"""你是一个专业的个人知识库助手，擅长基于用户的文档进行问答和分析。

{context}

【你的能力】
1. 语义搜索知识库 → 找到与用户问题最相关的文档片段
2. 阅读文档内容 → 了解文档的详细信息
3. 列出文档目录 → 查看知识库中有哪些文档
4. 构建索引 → 添加新文档后更新检索系统

【工作流程】
当用户提问时：
1. 先用 search_knowledge 检索相关文档片段
2. 如果片段不够详细，用 read_document 读取完整文档
3. 基于检索到的内容，组织清晰、准确的回答
4. 回答中标注信息来源（哪个文档）
5. 如果知识库中没有相关内容，诚实告知用户

【回答原则】
- 回答必须基于知识库中的文档内容，不要编造
- 引用文档内容时标注来源
- 如果知识库信息不足以回答，建议用户添加相关文档
- 用简洁清晰的语言组织答案

【知识库位置】{self.data_dir}/ 目录"""

        return prompt

    def run(self, task: str) -> str:
        """运行 Agent 回答问题（单轮）"""
        # 初始化状态和 Trace
        if self.enable_state:
            self.state.reset(task)
            self.state.status = AgentStatus.RUNNING
        if self.enable_trace:
            self.trace.start()

        system_prompt = self._build_system_prompt()
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]

        final_answer = ""

        for step in range(self.max_steps):
            if self.enable_state:
                self.state.step = step + 1
                self.state.status = AgentStatus.THINKING

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.registry.export_openai_tools(),
                    tool_choice="auto",
                    temperature=0.3,
                )
            except Exception as e:
                if self.enable_state:
                    self.state.status = AgentStatus.ERROR
                return f"调用模型时出错：{str(e)}"

            msg = response.choices[0].message
            self.messages.append(msg)

            # 提取 token usage
            token_usage = None
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            if not msg.tool_calls:
                final_answer = msg.content or "任务完成"
                if self.enable_state:
                    self.state.final_answer = final_answer
                    self.state.status = AgentStatus.FINISHED
                if self.enable_trace:
                    self.trace.record_final(final_answer)
                break

            # 模型在工具调用前的思考内容
            thought = msg.content or ""

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                if self.enable_state:
                    self.state.status = AgentStatus.CALLING_TOOL

                step_start = time.time()
                result = self.registry.execute(tool_name, tool_args)
                step_duration = (time.time() - step_start) * 1000

                if self.enable_state:
                    self.state.record_tool_call(tool_name, tool_args, result)

                if self.enable_trace:
                    self.trace.record_step(
                        step=step + 1,
                        thought=thought,
                        action=tool_name,
                        action_args=tool_args,
                        observation=result,
                        duration_ms=step_duration,
                        token_usage=token_usage,
                    )

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": str(result)
                })
        else:
            final_answer = "已达到最大步骤数，任务未完全完成。"
            if self.enable_state:
                self.state.final_answer = final_answer
                self.state.status = AgentStatus.MAX_STEPS

        return final_answer

    def chat(self, message: str) -> str:
        """多轮对话"""
        if not self.messages:
            return self.run(message)

        self.messages.append({"role": "user", "content": message})

        for step in range(self.max_steps):
            if self.enable_state:
                self.state.step = step + 1
                self.state.status = AgentStatus.THINKING

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.registry.export_openai_tools(),
                    tool_choice="auto",
                    temperature=0.3,
                )
            except Exception as e:
                return f"调用模型时出错：{str(e)}"

            msg = response.choices[0].message
            self.messages.append(msg)

            # 提取 token usage
            token_usage = None
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            if not msg.tool_calls:
                final_answer = msg.content or "好的"
                if self.enable_trace:
                    self.trace.record_final(final_answer)
                return final_answer

            thought = msg.content or ""

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                if self.enable_state:
                    self.state.status = AgentStatus.CALLING_TOOL

                step_start = time.time()
                result = self.registry.execute(tool_name, tool_args)
                step_duration = (time.time() - step_start) * 1000

                if self.enable_state:
                    self.state.record_tool_call(tool_name, tool_args, result)

                if self.enable_trace:
                    self.trace.record_step(
                        step=step + 1,
                        thought=thought,
                        action=tool_name,
                        action_args=tool_args,
                        observation=result,
                        duration_ms=step_duration,
                        token_usage=token_usage,
                    )

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": str(result)
                })

        return "已达到最大步骤数"

    def get_trace(self) -> Optional[dict]:
        """获取 Trace 数据（用于调试）"""
        return self.trace.export() if self.trace else None

    def get_state(self) -> Optional[dict]:
        """获取当前状态（用于调试）"""
        return self.state.to_dict() if self.state else None
