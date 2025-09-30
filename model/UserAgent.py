from langgraph.checkpoint.memory import MemorySaver
from typing import List
from langgraph.prebuilt import create_react_agent
import re
import json

SYSTEM_PROMPT = """
You are a user simulation Agent.
Here is the initial profile of the user you need to simulate:
{}
After this, you will receive an item description and its potentially applicable user profile each time.
You need to decide whether the user would interact with the item, based on analysis of the user information and the summary of previous simulations.
Your reply must follow the JSON format below:
{{"act": BOOL, "explain": string}}
Here, act is a boolean value indicating whether the user interacts with the item, and explain is a first-person explanation from the simulated user about this behavior.
Note: You must return PURE JSON only.
Think carefully before answering.
"""

FORWARD_PROMPT = """
Item description:
{}

Potentially applicable user profile:
{}
"""

POS_PROMPT = """
Great! You have correctly simulated the user’s behavior and provided a reasonable explanation.
"""

NEG_PROMPT = """
The user actually behaved in the opposite way to your prediction, which indicates that your understanding of the user's interests has some bias or omissions. 
You should adjust your understanding of the user based on this feedback, and then make another prediction and explanation.
"""


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("JSON NOT FOUND")

    json_str = match.group(0).strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON ERROR: {e}")


class UserAgent:
    def __init__(self, model, thread_id: str, tools: List, user_profile: str):
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": thread_id}}

        self.model = create_react_agent(
            model=model, tools=tools, checkpointer=self.memory
        )

        self.msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(user_profile)}]

    def run(self, is_print=False):
        resp = self.model.invoke({"messages": self.msgs}, config=self.config)

        if is_print:
            for msg in resp["messages"]:
                msg.pretty_print()

        try:
            content = resp["messages"][-1].content
            ans = extract_json(text=content)
            rec_id = ans["act"]
            explain = ans["explain"]
        except Exception as e:
            pass
        self.msgs=[]

        return ans

    def forward(self, item_desc: str, user_desc: str) -> tuple[int, str]:
        self.msgs.append(
            {
                "role": "user",
                "content": FORWARD_PROMPT.format(item_desc, user_desc),
            }
        )
        ans = self.run()

        return ans["act"], ans["explain"]

    def backward(self, feedback: bool):
        msg = POS_PROMPT if feedback else NEG_PROMPT

        self.msgs.append({"role": "user", "content": msg})

        self.run()
