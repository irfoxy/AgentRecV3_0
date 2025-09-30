from model.LLM import GPT
from model.UserAgent import UserAgent
import pandas as pd
from ast import literal_eval

base_path = "../data/ml100k/processed/"
MAX_ROW = 50
MAX_HISTORY = 10

COLD_START_PROMPT = """
Based on the following viewing history  
provide a preliminary profile of this user.  
Note that this profile should describe the user's interests and should not include any specific item information (such as titles or summaries).  
Your response should only include the profile itself.
"""


def cut_history(history_lst, is_random: bool = False):
    return history_lst[-MAX_HISTORY:]


def build_item_str(item_lst, item_dict):
    result = ""
    for item in item_lst:
        item_meta = item_dict[item]
        result += "\n" + f"id:{item} | " + item_meta
    return result


if __name__ == "__main__":
    # 读取数据
    behavior_df = pd.read_csv(base_path + "behavior.csv")
    candidate_df = pd.read_csv(base_path + "candidate.csv")
    user_meta_df = pd.read_csv(base_path + "user_meta.csv")
    item_meta_df = pd.read_csv(base_path + "item_meta.csv")
    predicted_df = pd.read_csv(base_path + "predicted.csv")

    # 转字典
    user_dict = user_meta_df.set_index("user_id")["metadata"].to_dict()
    item_dict = item_meta_df.set_index("item_id")["metadata"].to_dict()

    succ = 0
    count = 0

    for row in candidate_df.itertuples():
        if row.Index < 5:
            continue
        # 读取数据
        user_id = row.user_id
        candidate_lst = literal_eval(row.candidate)
        history_lst = cut_history(literal_eval(behavior_df.iloc[row.Index]["behavior"]))

        ground_truth = row.ground_truth
        history_str = build_item_str(history_lst, item_dict=item_dict)
        candidate_str = build_item_str(candidate_lst, item_dict=item_dict)
        predicted_str = str(predicted_df.iloc[row.Index]["predicted"])

        user_profile = GPT.invoke(COLD_START_PROMPT).content

        user_agent = UserAgent(
            model=GPT,
            thread_id=f"train_{row.Index}",
            tools=[],
            user_profile=user_profile,
        )

        history_lst=history_lst[:2]
        for item in history_lst:
            item_desc=item_dict[item]
            user_desc=""
            act,explain=user_agent.forward(item_desc=item_desc,user_desc=user_desc)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n\n")
            # print(act)
            # print(explain)
            user_agent.backward(feedback=False)
            # exit()
        exit()
        if row.Index >= MAX_ROW:
            print("TEST DONE")
            break
