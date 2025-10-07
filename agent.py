from .plugin import predict
from google.adk.agents import Agent
from google.adk.tools import Tool
import os
import logging
logging.basicConfig(level=logging.INFO)
from pydantic import BaseModel

predict_tool = Tool(
    name="predict_tool",
    description="Trả lời câu hỏi dựa trên hình ảnh.",
    func=predict,
    parameters={
        "image": {"type": "image", "description": "Ảnh cần phân tích"},
        "question": {"type": "string", "description": "Câu hỏi về ảnh"}
    }
)

class VQAResponse(BaseModel):
    answer: str
    confidence: float
    reasoning: str


vqa_agent = Agent(
    name="vqa_structured_agent",
    model="gemini-2.0-flash-thinking",
     description="""
        Agent nhận ảnh + câu hỏi, trả về JSON có cấu trúc:
        answer, confidence, reasoning.
    """,
    tools=[predict_tool],
    instruction="""
    Khi nhận ảnh + câu hỏi:
    1. Phân tích câu hỏi (reasoning)
    2. Nếu cần, gọi predict_tool
    3. Trả kết quả dưới dạng JSON:
        - answer: câu trả lời ngắn
        - confidence: độ tin cậy
        - reasoning: giải thích cách tìm ra câu trả lời
    """,
      output_type=VQAResponse, 
)

