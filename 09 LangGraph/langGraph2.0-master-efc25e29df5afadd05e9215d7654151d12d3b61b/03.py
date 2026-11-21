from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
import json

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# 执行图
result = graph.invoke({"foo": "", "bar":[]}, config)

print("=== 图执行结果 ===")
print(result)
print()

# 打印 checkpointer 中存储的所有 checkpoint
print("=== Checkpointer 存储内容 ===")
checkpoint_tuple = checkpointer.get_tuple(config)
if checkpoint_tuple:
    print(f"Thread ID: {checkpoint_tuple.config['configurable']['thread_id']}")
    print(f"Checkpoint ID: {checkpoint_tuple.checkpoint['id']}")
    print(f"Timestamp: {checkpoint_tuple.checkpoint['ts']}")
    print()
    print("=== Channel Values (实际状态) ===")
    channel_values = checkpoint_tuple.checkpoint["channel_values"]
    for key, value in channel_values.items():
        print(f"  {key}: {value}")
    print()
    print("=== 完整 Checkpoint ===")
    print(json.dumps(checkpoint_tuple.checkpoint, indent=2, ensure_ascii=False))
else:
    print("未找到存储的 checkpoint")