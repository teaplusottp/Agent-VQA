#from .plugin import predict
from google.adk.agents import Agent, Tool
from plugin import predict
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


predict_tool = Tool(
    name="predict_tool",
    description="Trả lời câu hỏi dựa trên hình ảnh và câu hỏi văn bản.",
    func=predict,
    parameters={
        "image": {
            "type": "image",
            "description": "Ảnh cần phân tích."
        },
        "question": {
            "type": "string",
            "description": "Câu hỏi về nội dung ảnh."
        }
    }
)


root_agent = Agent(
    name="vqa_agent_web",
    model="gemini-2.0-flash",
    description="Agent trả lời câu hỏi dựa trên hình ảnh người dùng gửi.",
    tools=[predict_tool],
    instruction="""
    Nếu người dùng gửi hình ảnh và một câu hỏi, hãy gọi công cụ `predict_tool`
    để trả về câu trả lời ngắn gọn và chính xác.
    """
)
