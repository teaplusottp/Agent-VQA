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

# Persistent memory
memory = Memory(storage_type="persistent", storage_path="../storage.db", max_history=50)

vqa_agent = Agent(
    name="vqa_agent",
    model="gemini-2.0-flash-thinking",
    tools=[predict_tool],
    output_type=VQAResponse,
    memory=memory,
    description="Agent trả lời VQA và lưu persistent memory.",
    instruction="""
    1. Phân tích câu hỏi + ảnh  
    2. Nếu cần, gọi predict_tool  
    3. Trả về JSON: answer, confidence, reasoning  
    4. Lưu câu hỏi + kết quả vào persistent memory
    """
)
