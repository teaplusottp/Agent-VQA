from google import genai
from google.adk.agents import Agent  # đổi từ genai -> adk

root_agent = Agent(
    name='greeting_agent',
    model='gemini-2.0-flash',
    description="An agent that can answer questions ",
    instruction="""
    Answer the question based on the image provided.
    """
)


