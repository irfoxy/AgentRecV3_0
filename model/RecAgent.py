from langgraph.checkpoint.memory import MemorySaver
from typing import List
from langgraph.prebuilt import create_react_agent
import json
from tools import extract_json
import re
import json

SYSTEM_PROMPT="""
You are an expert in {} recommendation.
You will receive the user's browsing history, including item IDs, item descriptions
Another recommendation system will provide several recommendation candidates for reference.
After that, you will select one item from a candidate list as the final recommendation.
Your response must follow the JSON format below:
{{"rec_id": INT, "explain": STRING}}
Note: You must return PURE JSON only.
Think carefully before answering.
"""

FORWARD_PROMPT="""
User history:
{}
Candidate list:
{}
This is another recommendation list provided by a different recommender system, for reference only:
{}
"""

RETRY_PROMPT="""
The user does not accept your recommendation.
The reasons are as follows:
{}
Based on the above information, analyze again and make a new recommendation.
"""

def extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("JSON NOT FOUND")
    
    json_str = match.group(0).strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON ERROR: {e}")

class RecAgent:
    def __init__(self,model,thread_id:str,tools:List,item_type:str):
        self.memory=MemorySaver()
        self.config = {"configurable": {"thread_id": thread_id}}

        self.model=create_react_agent(model=model,tools=tools,checkpointer=self.memory)
        
        self.msgs=[{"role":"system","content":SYSTEM_PROMPT.format(item_type)}] 
    
    def run(self,is_print=False):
        resp=self.model.invoke({"messages":self.msgs},config=self.config)

        if is_print:
            for msg in resp['messages']:
                msg.pretty_print()

        try:
            content=resp['messages'][-1].content
            ans=extract_json(text=content)
            rec_id=ans["rec_id"]
            explain=ans["explain"]
        except Exception as e:
            raise e
        
        return ans

    def forward(self,history_lst:str,candidate_lst:str,predicted:str)->tuple[int,str]:
        self.msgs.append({"role":"user","content":FORWARD_PROMPT.format(history_lst,candidate_lst,predicted)})
        ans=self.run()

        return ans['rec_id'],ans['explain']

    def retry(self,feed_back:str):
        self.msgs.append({"role":"user","content":RETRY_PROMPT.format(feed_back)})
        ans=self.run()

        return ans



    