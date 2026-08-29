"""单元测试骨架（M2/M3 填充）。运行：pip install pytest && pytest tests/

建议覆盖：
- edit 工具：0 次匹配 / 多次匹配 / 唯一匹配成功 / replace_all
- dispatch 容错链：非法 JSON、缺 required 参数、工具内抛异常 → 都是结构化 ToolResult
- bash：超时、输出截断标记、退出码回传、交互命令拦截
- safety：路径穿越（../）、敏感路径、危险命令 fail-closed
- agent：mock LLMClient 验证主出口、max_turns 兜底、tool 结果按 id 回写
"""
import pytest  # noqa: F401


def test_placeholder():
    assert True
