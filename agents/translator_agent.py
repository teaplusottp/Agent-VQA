from google.adk.agents import Agent
from google.adk.memory import Memory

# Dùng chung storage
memory = Memory(storage_type="persistent", storage_path="../storage.db", max_history=50)

translator_agent = Agent(
    name="translator_agent",
    model="gemini-2.0-flash",
    memory=memory,
    description="Dịch output JSON của các agent khác sang ngôn ngữ khác.",
    instruction="""
    1. Lấy output JSON từ memory  
    2. Dịch text sang ngôn ngữ user yêu cầu (ví dụ tiếng Anh -> tiếng Việt)  
    3. Trả về JSON: translated_output
    """
)
