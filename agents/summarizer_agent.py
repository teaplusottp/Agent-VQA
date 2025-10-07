from google.adk.agents import Agent
from google.adk.memory import Memory

memory = Memory(storage_type="persistent", storage_path="../storage.db", max_history=50, stateful=True)

summarizer_agent = Agent(
    name="summarizer_agent",
    model="gemini-2.0-flash",
    memory=memory,
    description="Tổng hợp kết quả nhiều câu hỏi/ảnh, giữ trạng thái riêng",
    instruction="""
    1. Lấy lịch sử VQA từ memory
    2. Tóm tắt kết quả, giữ context summary
    3. Lưu summary vào memory
    """
)
