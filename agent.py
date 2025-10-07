#from .plugin import predict
from google.adk.agents import Agent, Tool
import os
import logging
logging.basicConfig(level=logging.INFO)

from google import genai

client = genai.Client()

def predict(image_path: str, question: str, question_type: int = 0):
    """
    Dùng model Gemini (hoặc model multimodal khác) để trả lời câu hỏi về ảnh.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                {"role": "user", "parts": [
                    {"text": question},
                    {"inline_data": {
                        "mime_type": "image/png", 
                        "data": open(image_path, "rb").read()
                    }}
                ]}
            ]
        )
        return response.text

    except Exception as e:
        return f"❌ Lỗi: {e}"


# === Tool: predict_tool ===
predict_tool = Tool(
    name="predict_tool",
    description="Phân tích hình ảnh và trả lời câu hỏi về nội dung ảnh.",
    func=predict,
    parameters={
        "image": {"type": "image", "description": "Ảnh cần phân tích."},
        "question": {"type": "string", "description": "Câu hỏi liên quan đến ảnh."}
    }
)


# === Agent: vqa_tool_agent ===
root_agent = Agent(
    name="vqa_tool_agent",
    model="gemini-2.0-flash",
    description="Agent nhận ảnh + câu hỏi và dùng tool để trả lời.",
    tools=[predict_tool],
    instruction="""
    Nếu người dùng gửi hình ảnh, hãy **tự động gọi công cụ predict_tool** để phân tích nội dung ảnh.
    Nếu người dùng chỉ hỏi về văn bản, hãy tự trả lời bằng ngôn ngữ tự nhiên.
    Khi dùng predict_tool, truyền cả ảnh và câu hỏi của người dùng.
    """
)
