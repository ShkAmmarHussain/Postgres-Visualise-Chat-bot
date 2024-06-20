import os
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import initialize_agent
from llm import chat_llm
from dotenv import load_dotenv

load_dotenv()

def _handle_error(error) -> str:
    return str(error)[:50]

# Primary Assistant
@tool
class ToProvideInformationgAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle information gathering."""

    request: str = Field(
        description="Question that the user asked."
    )

class chat_agent():
    def __init__(self):
        self.agent = None
        self.chat_assistant = chat_llm()
        self.df = self.chat_assistant.create_llm_chain(
            model_id=os.getenv("LLM_MODEL"),
            provider=os.getenv("LLM_PROVIDER")
        )
    
    def agent_creation(self):
        prim_tools = [ToProvideInformationgAssistant]


        self.agent = initialize_agent(
            prim_tools,
            self.df.llm.query,
            agent="zero-shot-react-description",
            # agent = "structured-chat-zero-shot-react-description",
            verbose=True,
            max_iterations = 4,
            handle_tool_errors = _handle_error
        #     agent_kwargs={
        #         'prefix': PREFIX,
        #         # 'format_instructions': FORMAT_INSTRUCTIONS,
        #         'suffix': SUFFIX
        #    
        )