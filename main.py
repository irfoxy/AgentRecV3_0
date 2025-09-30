from model.RecAgent import RecAgent
import pandas as pd
from ast import literal_eval
from model.LLM import GPT
from model.tools import search

base_path = "../data/ml100k/processed/"
MAX_ROW = 200
MAX_HISTORY=10

def cut_history(history_lst,is_random:bool=False):
    return history_lst[-MAX_HISTORY:]

def build_item_str(item_lst,item_dict):
    result=""
    for item in item_lst:
        item_meta=item_dict[item]
        result+='\n'+f"id:{item} | "+item_meta
    return result

if __name__ == "__main__":
    # 读取数据
    behavior_df = pd.read_csv(base_path + "behavior.csv")
    candidate_df = pd.read_csv(base_path + "candidate.csv")
    user_meta_df = pd.read_csv(base_path + "user_meta.csv")
    item_meta_df = pd.read_csv(base_path + "item_meta.csv")
    predicted_df=pd.read_csv(base_path+"predicted.csv")

    # 转字典
    user_dict = user_meta_df.set_index("user_id")["metadata"].to_dict()
    item_dict = item_meta_df.set_index("item_id")["metadata"].to_dict()

    succ=0
    count=0

    for row in candidate_df.itertuples():
        if row.Index<5:
            continue
        # 读取数据
        user_id = row.user_id
        candidate_lst = literal_eval(row.candidate)
        history_lst=cut_history(literal_eval(behavior_df.iloc[row.Index]['behavior']))

        ground_truth = row.ground_truth
        history_str=build_item_str(history_lst,item_dict=item_dict)
        candidate_str=build_item_str(candidate_lst,item_dict=item_dict)
        predicted_str=str(predicted_df.iloc[row.Index]['predicted'])
        
        # 创建rec agent
        rec_agent = RecAgent(
            model=GPT, thread_id=f"user_{row.Index}", tools=[search], item_type="movie"
        )

        # print("start rec")

        rec_id,explain=rec_agent.forward(history_lst=history_str,candidate_lst=candidate_str,predicted=predicted_str)

        print(f"{rec_id} {ground_truth}")
        # print(explain)
        count+=1
        if rec_id==ground_truth:
            succ+=1
        hit_1=succ/count

        print(f"HIT@1: {hit_1}")
        with open("../result/hit.txt", "a", encoding="utf-8") as f:
            f.write(str(hit_1)+'\n')
        
        if row.Index>=MAX_ROW:
            print("TEST DONE")
            break

        
