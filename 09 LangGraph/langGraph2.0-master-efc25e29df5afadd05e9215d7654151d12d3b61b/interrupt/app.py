"""
LangGraph 中断与恢复演示 - Checkpointer 状态追踪
"""

import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Literal
import json

BASE_DIR = Path(__file__).resolve().parent
from dotenv import load_dotenv
load_dotenv(dotenv_path=BASE_DIR / ".env")

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langchain.messages import HumanMessage, AIMessage, AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


# ============================================================
# 状态定义
# ============================================================
class ApprovalState(TypedDict):
    task_id: str
    task_content: str
    status: Optional[Literal["pending", "approved", "rejected"]]
    final_result: str
    messages: Annotated[list[AnyMessage], operator.add]


# ============================================================
# 节点
# ============================================================
def process_task(state: ApprovalState):
    return {
        "messages": [AIMessage(content=f"已处理任务：{state['task_content']}")],
        "final_result": f"已处理任务：{state['task_content']}",
    }


def wait_approval(state: ApprovalState) -> Command[Literal["after_approval", "cancel_node"]]:
    decision = interrupt({
        "question": "请确认是否批准该任务？",
        "task_id": state["task_id"],
        "task_content": state["task_content"],
    })
    return Command(goto="after_approval" if decision else "cancel_node")


def after_approval(state: ApprovalState):
    return {
        "status": "approved",
        "messages": [AIMessage(content=f"任务 {state['task_id']} 已通过审批")],
        "final_result": f"任务 {state['task_id']} 已通过审批",
    }


def cancel_node(state: ApprovalState):
    return {
        "status": "rejected",
        "messages": [AIMessage(content=f"任务 {state['task_id']} 已被拒绝")],
        "final_result": f"任务 {state['task_id']} 已被拒绝",
    }


# ============================================================
# 构建图
# ============================================================
def create_graph():
    workflow = StateGraph(ApprovalState)
    workflow.add_node("process_task", process_task)
    workflow.add_node("wait_approval", wait_approval)
    workflow.add_node("after_approval", after_approval)
    workflow.add_node("cancel_node", cancel_node)
    workflow.add_edge(START, "process_task")
    workflow.add_edge("process_task", "wait_approval")
    workflow.add_edge("after_approval", END)
    workflow.add_edge("cancel_node", END)
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer), checkpointer


graph, checkpointer = create_graph()


# ============================================================
# 打印 Checkpoint 状态
# ============================================================
def print_checkpoint(thread_id: str, title: str):
    """打印指定 thread 的 checkpoint 状态"""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = checkpointer.get_tuple(config)
    
    print(f"\n{'='*60}")
    print(f"[{title}]")
    print(f"{'='*60}")
    
    if not checkpoint:
        print("  未找到 checkpoint")
        return
    
    cv = checkpoint.checkpoint["channel_values"]
    print(f"  Thread ID: {thread_id}")
    print(f"  Checkpoint ID: {checkpoint.checkpoint['id']}")
    print(f"  Timestamp: {checkpoint.checkpoint['ts']}")
    print(f"\n  [Channel Values 状态]")
    for key, value in cv.items():
        if key == "messages":
            print(f"    messages: [{len(value)} 条消息]")
            for i, msg in enumerate(value):
                print(f"      [{i}] {msg.type}: {msg.content[:50]}...")
        else:
            print(f"    {key}: {value}")
    print(f"{'='*60}\n")


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="LangGraph Interrupt Demo")
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


class TaskRequest(BaseModel):
    task_id: str
    task_content: str


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    reason: Optional[str] = None


@app.get("/")
async def read_root():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/start-task")
async def start_task(request: TaskRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 初始状态
    initial_state = {
        "task_id": request.task_id,
        "task_content": request.task_content,
        "status": "pending",
        "final_result": "",
        "messages": [HumanMessage(content=request.task_content)],
    }

    # 执行到中断点
    result = graph.invoke(initial_state, config)

    # 打印中断前的 checkpoint
    print_checkpoint(thread_id, "启动任务后 - 中断前")

    # 检查中断信息
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        return {
            "success": True,
            "thread_id": thread_id,
            "status": "waiting_approval",
            "message": "任务已暂停，等待审批",
        }

    return {
        "success": True,
        "thread_id": thread_id,
        "status": "completed",
        "result": result,
    }


@app.post("/api/approve")
async def approve_task(request: ApprovalRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    # 打印恢复前的 checkpoint
    print_checkpoint(request.thread_id, "恢复流程前 - 检查点状态")

    # 恢复执行
    result = graph.invoke(Command(resume=request.approved), config=config)

    # 打印恢复后的 checkpoint
    print_checkpoint(request.thread_id, "恢复流程后 - 最终状态")

    return {
        "success": True,
        "status": "completed",
        "result": result,
    }


@app.get("/api/status/{thread_id}")
async def get_status(thread_id: str):
    print_checkpoint(thread_id, "查询状态")
    return {"success": True, "thread_id": thread_id}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("LangGraph Checkpointer 状态追踪演示")
    print("="*60)
    print("访问 http://localhost:8000 进行演示")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
