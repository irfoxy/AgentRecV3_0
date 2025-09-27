from model.RecAgent import RecAgent
import pandas as pd
from ast import literal_eval

base_path="../data/ml100k/processed/"

if __name__=="__main__":
    behavior_df=pd.read_csv(base_path+"behavior.csv")
    candidate_df=pd.read_csv(base_path+"candidate.csv")
    user_meta_df=pd.read_csv(base_path+"user_meta.csv")
    item_meta_df=pd.read_csv(base_path+"item_meta.csv")
