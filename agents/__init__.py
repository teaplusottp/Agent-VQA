from .vqa_agent import vqa_agent
from .summarizer_agent import summarizer_agent
from .translator_agent import translator_agent

# Root agent list để adk web load
root_agents = [vqa_agent, summarizer_agent, translator_agent]
