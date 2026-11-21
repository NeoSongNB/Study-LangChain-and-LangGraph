# LangGraph 中断与恢复演示

本演示展示了 LangGraph 的**中断机制** - 当流程需要人工审批时暂停，用户确认后恢复执行。

## 文件结构

```
interrupt/
├── app.py          # 后端服务 (FastAPI + LangGraph)
├── index.html      # 前端界面
├── requirements.txt # Python 依赖
└── README.md       # 说明文档
```

## 快速开始

### 1. 安装依赖

```bash
cd interrupt
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python app.py
```

### 3. 打开浏览器

访问 http://localhost:8000

## 使用方法

1. **提交任务**：在表单中填写任务 ID 和内容
   - 任务 ID 包含 `need_approval` 会触发审批流程
   - 例如：`task-need_approval-001`

2. **等待中断**：流程执行到审批节点时会暂停

3. **确认/拒绝**：在界面上点击"确认通过"或"拒绝"

4. **查看结果**：流程恢复执行，显示最终结果

## 流程图

```
START → process_task → [判断] → wait_approval → after_approval → END
                              ↓
                         (中断等待用户确认)
                              ↓
                         用户点击确认/拒绝
                              ↓
                         继续执行
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/start-task` | POST | 启动任务 |
| `/api/approve` | POST | 审批确认 |
| `/api/status/{thread_id}` | GET | 查询状态 |

## 依赖说明

- **langgraph**: 核心图框架
- **fastapi**: Web 框架
- **uvicorn**: ASGI 服务器

