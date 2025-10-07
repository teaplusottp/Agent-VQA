from .plugin import predict
from google.adk.agents import Agent
from google.adk.tools import Tool
import os
import logging
logging.basicConfig(level=logging.INFO)

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

vqa_agent = Agent(
    name="vqa_litemm_agent",
    model="gemini-2.0-flash-thinking",
    description="""
        Agent nhận ảnh + câu hỏi, tự suy nghĩ trước khi gọi tool predict_tool.
        Nếu ảnh rõ ràng, agent trực tiếp trả lời. Nếu cần, gọi tool.
    """,
    tools=[predict_tool],
    instruction="""
    Khi user gửi ảnh và câu hỏi:
    1. Phân tích câu hỏi, xác định thông tin cần tìm.
    2. Nếu model có thể trả lời dựa trên reasoning, trả lời trực tiếp.
    3. Nếu cần thông tin chi tiết từ ảnh, gọi tool predict_tool.
    4. Kết hợp reasoning + output từ tool để trả về câu trả lời chính xác.
    """
)

