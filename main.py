from model.RecAgent import RecAgent
import pandas as pd
from ast import literal_eval
from model.LLM import GPT
from model.tools import search
from model.UserAgent import UserAgent
import random
from typing import List, Tuple
from tqdm import tqdm

base_path = "../data/ml100k/processed/"
MAX_ROW = 200
MAX_HISTORY = 10
MAX_TRY = 2
MAX_ITEM_IDX = 1681

COLD_START_PROMPT = """
Based on the following viewing history  
{}
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


def build_train_lst(
    history_lst: List[int], behavior_lst: List[int], n: int, max_item_idx: int = 1681
) -> Tuple[List[int], List[bool]]:

    sample_lst, label_lst = [], []

    # 建立负样本候选池（排除行为里的物品）
    behavior_set = set(behavior_lst)
    neg_pool = [i for i in range(0, max_item_idx + 1) if i not in behavior_set]

    for _ in range(n):
        if random.choice([True, False]) and history_lst:
            # 正样本
            item = random.choice(history_lst)
            sample_lst.append(item)
            label_lst.append(True)
        else:
            # 负样本，从 neg_pool 采样
            item = random.choice(neg_pool)
            sample_lst.append(item)
            label_lst.append(False)

    return sample_lst, label_lst


def train_agent(
    user_agent: UserAgent, sample_lst: list, label_lst: list, item_dict: dict
):
    for idx, sample in tqdm(enumerate(sample_lst), total=len(sample_lst)):
        act, explain = user_agent.forward(item_desc=item_dict[sample], user_desc="")
        feedback = act == label_lst[idx]
        user_agent.backward(feedback=feedback)


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
        thread_id = f"user_{row.Index}"

        if row.Index < 5:
            continue
        # 读取数据
        user_id = row.user_id
        candidate_lst = literal_eval(row.candidate)
        behavior_lst = literal_eval(behavior_df.iloc[row.Index]["behavior"])
        history_lst = cut_history(behavior_lst)

        ground_truth = row.ground_truth
        history_str = build_item_str(history_lst, item_dict=item_dict)
        candidate_str = build_item_str(candidate_lst, item_dict=item_dict)
        predicted_str = str(predicted_df.iloc[row.Index]["predicted"])

        # 创建rec agent
        rec_agent = RecAgent(
            model=GPT, thread_id=thread_id, tools=[search], item_type="movie"
        )

        print("start rec")
        rec_id, rec_explain = rec_agent.forward(
            history_lst=history_str,
            candidate_lst=candidate_str,
            predicted=predicted_str,
        )

        # user_profile = GPT.invoke(COLD_START_PROMPT.format(history_str)).content
        # user_agent = UserAgent(
        #     model=GPT, thread_id=thread_id, tools=[], user_profile=user_profile
        # )

        # sample_lst, label_lst = build_train_lst(
        #     history_lst=history_lst,
        #     n=len(history_lst),
        #     behavior_lst=behavior_lst,
        # )
        # print("---START TRAINING---")
        # train_agent(
        #     user_agent=user_agent,
        #     sample_lst=sample_lst,
        #     label_lst=label_lst,
        #     item_dict=item_dict,
        # )

        # retry_count = 0
        # print("start loop")
        # while True:
        #     act, feedback_explain = user_agent.forward(
        #         item_desc=item_dict[rec_id], user_desc=""
        #     )
        #     retry_count += 1
        #     if act or retry_count >= MAX_TRY:
        #         # print("ACCEPTED")
        #         break
        #     else:
        #         # print("REJECTED")
        #         rec_id, rec_explain = rec_agent.retry(feed_back=feedback_explain)

        print(f"{rec_id} {ground_truth}")
        count += 1
        if rec_id == ground_truth:
            succ += 1
        hit_1 = succ / count

        print(f"HIT@1: {hit_1}")
        with open("../result/hit.txt", "a", encoding="utf-8") as f:
            f.write(str(hit_1) + "\n")

        if row.Index >= MAX_ROW:
            print("TEST DONE")
            break
