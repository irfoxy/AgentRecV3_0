from LLM import GPT
from langgraph.checkpoint.memory import MemorySaver
from typing import List

class RecAgent:
    def __init__(self,thread_id:str,tools:List):
        self.llm=GPT
        self.memory=MemorySaver()
        self.config={"condigurable":{"thread_id":thread_id}}

    def forward(self,history:str,candidate:str)->int:
        pass