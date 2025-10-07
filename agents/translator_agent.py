from google.adk.agents import Agent
from google.adk.memory import Memory

memory = Memory(storage_type="persistent", storage_path="../storage.db", max_history=50, stateful=True)

translator_agent = Agent(
    name="translator_agent",
    model="gemini-2.0-flash",
    memory=memory,
    description="Dịch output JSON của các agent khác, stateful",
    instruction="""
    1. Lấy output JSON từ memory
    2. Dịch sang ngôn ngữ user yêu cầu
    3. Lưu translated_output vào memory
    """
)
