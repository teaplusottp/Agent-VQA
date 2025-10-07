from google.adk.agents import Agent
from google.adk.tools import Tool
from google.adk.memory import Memory
from pydantic import BaseModel
from plugin import predict

# === Tool ===
predict_tool = Tool(
    name="predict_tool",
    description="Trả lời câu hỏi dựa trên hình ảnh.",
    func=predict,
    parameters={
        "image": {"type": "image", "description": "Ảnh cần phân tích"},
        "question": {"type": "string", "description": "Câu hỏi về ảnh"}
    }
)

# === Output structured ===
class VQAResponse(BaseModel):
    answer: str
    confidence: float
    reasoning: str

# === Persistent Memory ===
memory = Memory(
    storage_type="persistent",   # khác với session
    storage_path="./storage.db", # SQLite db
    max_history=50               # lưu tối đa 50 turn
)

# === Agent với Persistent Storage ===
vqa_agent = Agent(
    name="vqa_persistent_agent",
    model="gemini-2.0-flash-thinking",
    tools=[predict_tool],
    output_type=VQAResponse,
    memory=memory,
    description="""
    Agent lưu trữ lâu dài câu hỏi + ảnh + kết quả.
    Khi restart server, vẫn nhớ lịch sử của từng user.
    """,
    instruction="""
    1. Khi nhận câu hỏi + ảnh, agent kiểm tra memory.
    2. Nếu cần, gọi predict_tool.
    3. Kết hợp reasoning + memory để trả lời.
    4. Lưu câu hỏi + kết quả vào persistent storage.
    """
)
