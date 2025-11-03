# -*- coding: utf-8 -*-
"""The main entry point of the plan example."""
import asyncio
import os

from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.plan import PlanNotebook
from agentscope.tool import (
    Toolkit,
    execute_shell_command,
    execute_python_code,
    write_text_file,
    insert_text_file,
    view_text_file,
)


async def main() -> None:
    """The main entry point for the plan example."""
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_shell_command)
    toolkit.register_tool_function(execute_python_code)
    toolkit.register_tool_function(write_text_file)
    toolkit.register_tool_function(insert_text_file)
    toolkit.register_tool_function(view_text_file)

    agent = ReActAgent(
        name="Friday",
        sys_prompt="""You're a helpful assistant named Friday.

# Target
Your target is to finish the given task with careful planning.
""",  # noqa
        model=DashScopeChatModel(
            model_name="qwen3-max-preview",
            api_key=os.environ["DASHSCOPE_API_KEY"],
        ),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        enable_meta_tool=True,
        auto_plan=True,
        max_iters=50,
        compress_tool_history=True,
    )
    user = UserAgent(name="user")
    agent.model = DashScopeChatModel(
            model_name="qwen3-max-preview",
            api_key=os.environ["DASHSCOPE_API_KEY"],
        )

    msg = Msg(
        "user",
        "Review the recent changes in AgentScope GitHub repository "
        "over the past month.",
        "user",
    )
    while True:
        msg = await agent(msg)
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break


asyncio.run(main())
