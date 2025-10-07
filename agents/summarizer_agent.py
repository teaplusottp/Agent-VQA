from google.adk.agents import Agent
from google.adk.memory import Memory

# Dùng chung storage
memory = Memory(storage_type="persistent", storage_path="../storage.db", max_history=50)

summarizer_agent = Agent(
    name="summarizer_agent",
    model="gemini-2.0-flash",
    memory=memory,
    description="Tổng hợp kết quả nhiều câu hỏi/ảnh từ VQA agent.",
    instruction="""
    1. Lấy lịch sử hỏi đáp từ memory (vqa_agent)  
    2. Tóm tắt các câu trả lời thành report ngắn gọn  
    3. Trả về JSON: summary
    """
)
