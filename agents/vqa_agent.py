from google.adk.agents import Agent
from google.adk.tools import Tool
from google.adk.memory import Memory
from pydantic import BaseModel
from plugin import predict

# Tool VQA
predict_tool = Tool(
    name="predict_tool",
    description="Trả lời câu hỏi dựa trên hình ảnh.",
    func=predict,
    parameters={
        "image": {"type": "image", "description": "Ảnh cần phân tích"},
        "question": {"type": "string", "description": "Câu hỏi về ảnh"}
    }
)

# Output structured
class VQAResponse(BaseModel):
    answer: str
    confidence: float = 0.95
    reasoning: str

# Stateful persistent memory
memory = Memory(
    storage_type="persistent",
    storage_path="../storage.db",
    max_history=50,
    stateful=True  # đây là key để agent giữ trạng thái riêng
)

# Stateful Agent
vqa_agent = Agent(
    name="vqa_stateful_agent",
    model="gemini-2.0-flash-thinking",
    tools=[predict_tool],
    output_type=VQAResponse,
    memory=memory,
    description="Agent VQA có trạng thái riêng, multi-turn reasoning, persistent memory.",
    instruction="""
    1. Nhớ history + context user trong memory riêng
    2. Khi nhận câu hỏi mới, kiểm tra state + history
    3. Sử dụng reasoning + tool nếu cần để trả lời
    4. Lưu câu hỏi + kết quả + reasoning vào memory
    """
)
