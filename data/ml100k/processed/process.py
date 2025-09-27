import csv

# 文件路径
user_file = "../raw/u.user"
item_file = "../raw/u.item"
genre_file = "../raw/u.genre"  # 这是存储电影类型的文件

# 输出文件
user_meta_file = "./user_meta.csv"
item_meta_file = "./item_meta.csv"

# 读取电影类别文件 u.genre，获取类别名称
def read_genre_data(genre_file):
    genres = []
    with open(genre_file, 'r') as file:
        genres = [line.strip() for line in file]
    return genres

# 读取用户文件 u.user，获取用户信息
def read_user_data(user_file):
    user_data = {}
    with open(user_file, 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            user_id = int(parts[0])  # 获取用户 ID
            age = parts[1]  # 获取用户年龄
            sex = parts[2]  # 获取用户性别
            occupation = parts[3]  # 获取用户职业
            zip_code = parts[4]  # 获取用户邮政编码
            
            # 生成用户元数据字符串
            metadata = f"Age: {age}, Sex: {sex}, Occupation: {occupation}, Zip: {zip_code}"
            
            # 将元数据存储
            user_data[user_id] = metadata
    return user_data

# 读取物品文件 u.item，并根据 u.genre 生成每个物品的元数据
def read_item_data(item_file, genres):
    item_data = {}
    with open(item_file, 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split('|')
            item_id = int(parts[0])  # 获取物品 ID
            title = parts[1]  # 获取电影标题
            release_date = parts[2]  # 获取电影发布日期
            imdb_url = parts[4]  # 获取电影 IMDB 链接
            genre_flags = parts[5:]  # 获取电影的类型标签（0 或 1）
            
            # 根据标记选择相应的类别名称
            selected_genres = [genres[i] for i, flag in enumerate(genre_flags) if flag == '1']
            
            # 生成元数据字符串，调整格式为：Title + | + Genres
            metadata = f"Title: {title} | Genres: {', '.join(selected_genres)}"
            
            # 将元数据存储
            item_data[item_id] = metadata
    return item_data

# 写入用户元数据到 CSV 文件
def write_user_meta(user_data, user_meta_file):
    with open(user_meta_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'metadata'])
        for user_id, metadata in user_data.items():
            writer.writerow([user_id, metadata])

# 写入物品元数据到 CSV 文件
def write_item_meta(item_data, item_meta_file):
    with open(item_meta_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['item_id', 'metadata'])
        for item_id, metadata in item_data.items():
            writer.writerow([item_id, metadata])

# 主程序
def main():
    # 读取类别数据
    genres = read_genre_data(genre_file)
    
    # 读取用户数据
    user_data = read_user_data(user_file)
    
    # 读取物品数据
    item_data = read_item_data(item_file, genres)
    
    # 写入到 CSV 文件
    write_user_meta(user_data, user_meta_file)
    write_item_meta(item_data, item_meta_file)
    print(f"User metadata saved to {user_meta_file}")
    print(f"Item metadata saved to {item_meta_file}")

# 运行主程序
if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import argparse
# import random
# from typing import List, Dict, Tuple

# import pandas as pd


# def build_user_histories(
#     udata_path: str,
# ) -> Tuple[Dict[int, List[Tuple[int, int, int]]], List[int]]:
#     """
#     读取 u.data -> 返回:
#       user2events[user] = [(item, rating, ts), ...]（已按时间升序）
#       all_items = 所有出现过的 item 列表（去重，升序）
#     """
#     # u.data: user_id \t item_id \t rating \t timestamp
#     df = pd.read_csv(
#         udata_path,
#         sep=r"\s+|\t",
#         engine="python",
#         header=None,
#         names=["user_id", "item_id", "rating", "timestamp"],
#     )

#     # 按时间排序
#     df = df.sort_values(["user_id", "timestamp"], ascending=[True, True])

#     user2events: Dict[int, List[Tuple[int, int, int]]] = {}
#     for row in df.itertuples(index=False):
#         uid = int(getattr(row, "user_id"))
#         iid = int(getattr(row, "item_id"))
#         rating = int(getattr(row, "rating"))
#         ts = int(getattr(row, "timestamp"))
#         user2events.setdefault(uid, []).append((iid, rating, ts))

#     all_items = sorted(df["item_id"].unique().tolist())
#     return user2events, all_items


# def truncate_by_last_high(events: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
#     """
#     给定该用户的按时间升序序列 events = [(item, rating, ts), ...]
#     找到**最后一个** rating ∈ {4,5} 的位置 idx_last_high，
#     丢弃该位置及其后的所有事件，返回剩余前缀。
#     若不存在 rating∈{4,5}，则不截断，返回全部。
#     """
#     last_idx = -1
#     for idx, (_, rating, _) in enumerate(events):
#         if rating >= 4:
#             last_idx = idx
#     if last_idx == -1:
#         # 没有高分，则不截断
#         return events
#     # 丢弃该位置及之后
#     return events[:last_idx]


# def build_pos_from_log(truncated_events: List[Tuple[int, int, int]]) -> List[int]:
#     """
#     在截断后的日志中，收集 rating ∈ {4,5} 的 item 作为 pos。
#     """
#     return [itm for (itm, rating, _) in truncated_events if rating >= 4]


# def sample_neg_items(
#     all_items: List[int],
#     user_interacted_set: set,
#     num_neg: int,
#     rng: random.Random,
# ) -> List[int]:
#     """
#     从用户未交互集合里随机采样 num_neg 个不同的 item。
#     如果可用数量不足，则全取。
#     """
#     candidates = [it for it in all_items if it not in user_interacted_set]
#     if len(candidates) <= num_neg:
#         rng.shuffle(candidates)
#         return candidates
#     return rng.sample(candidates, num_neg)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_dir", type=str, default="../raw", help="原始 MovieLens100K 目录（含 u.data）")
#     parser.add_argument("--output_dir", type=str, default="./", help="输出目录（生成 train.csv）")
#     parser.add_argument("--num_neg", type=int, default=20, help="每个用户采样的负样本数量")
#     parser.add_argument("--seed", type=int, default=42, help="随机种子")
#     args = parser.parse_args()

#     rng = random.Random(args.seed)

#     udata_path = os.path.join(args.input_dir, "u.data")
#     if not os.path.exists(udata_path):
#         raise FileNotFoundError(f"未找到 {udata_path}，请确认 --input_dir 设置正确。")

#     os.makedirs(args.output_dir, exist_ok=True)
#     out_csv = os.path.join(args.output_dir, "train.csv")

#     # 1) 读取并整理用户事件
#     user2events, all_items = build_user_histories(udata_path)

#     rows = []
#     for uid, events in user2events.items():
#         # 该用户所有交互过的物品集合（用于 neg 采样去重）
#         interacted_items = {itm for (itm, _, _) in events}

#         # 2) 截断：丢弃“最后一个高分(4或5)”及其之后
#         truncated = truncate_by_last_high(events)

#         # 3) 构建 log（仅 item，按时间升序）
#         log = [itm for (itm, _, _) in truncated]

#         # 4) 构建 pos（截断后的 log 中 rating ∈ {4,5} 的 item）
#         pos = build_pos_from_log(truncated)

#         # 5) 采样 neg（从用户未交互集合中随机）
#         neg = sample_neg_items(all_items, interacted_items, args.num_neg, rng)

#         rows.append(
#             {
#                 "user_id": uid,
#                 # 直接以 Python 列表字符串存储，后续可用 ast.literal_eval 读取
#                 "log_seq": str(log),
#                 "pos": str(pos),
#                 "neg": str(neg),
#             }
#         )

#     pd.DataFrame(rows, columns=["user_id", "log_seq", "pos", "neg"]).to_csv(out_csv, index=False, encoding="utf-8")
#     print(f"已生成: {out_csv}")
#     print(f"用户数: {len(rows)} | 物品数: {len(all_items)}")


# if __name__ == "__main__":
#     main()


# import os
# import json
# import random
# import pandas as pd

# RAW_DIR = "../raw"   # 输入目录
# OUT_DIR = "./"       # 输出目录
# SEED = 2024
# random.seed(SEED)

# def read_ml100k(raw_dir):
#     # u.data: user_id, item_id, rating, timestamp (tab 分隔)
#     cols = ["user_id", "item_id", "rating", "timestamp"]
#     df = pd.read_csv(
#         os.path.join(raw_dir, "u.data"),
#         sep="\t",
#         names=cols,
#         engine="python"
#     )
#     # u.item: 取第一列的 movie_id 作为全集 item 集合
#     # 注意：u.item 用 | 分隔，第一列为电影ID
#     item_df = pd.read_csv(
#         os.path.join(raw_dir, "u.item"),
#         sep="|",
#         header=None,
#         encoding="latin-1",
#         engine="python"
#     )
#     item_df = item_df.rename(columns={0: "item_id"})
#     all_items = set(item_df["item_id"].tolist())
#     return df, all_items

# def make_behavior_and_candidate(df, all_items):
#     """
#     生成 behavior.csv 与 candidate.csv
#     规则：
#       - ground_truth: 用户最近一次评分为 4/5 的 item
#       - behavior: 该用户除 ground_truth 外其它评分为 4/5 的 item，按时间升序
#       - 若用户没有 4/5 或只有 ground_truth 而无其它 4/5，则剔除该用户
#       - candidate: 对每个用户构造 ground_truth + 19 个未交互 item（打乱）
#     """
#     # 仅保留评分 >=4 的记录做高分筛选
#     high_df = df[df["rating"] >= 4].copy()
#     if high_df.empty:
#         return pd.DataFrame(columns=["user_id","behavior","ground_truth"]), pd.DataFrame(columns=["user_id","candidate","ground_truth"])

#     # 为了找最近一次 4/5，按 timestamp 升序后再 groupby 取最后一个
#     high_df = high_df.sort_values(["user_id", "timestamp"])
#     # 最近一次 4/5 的行作为 ground_truth
#     last_high = high_df.groupby("user_id").tail(1)
#     gt_map = dict(zip(last_high["user_id"], last_high["item_id"]))
#     gt_time_map = dict(zip(last_high["user_id"], last_high["timestamp"]))

#     # 构造 behavior：该用户其它 4/5（去掉 ground_truth），按时间升序
#     behavior_rows = []
#     candidate_rows = []

#     # 用户全交互集合（任何评分）用于确定“未交互”
#     interacted = df.groupby("user_id")["item_id"].apply(set).to_dict()

#     for u, u_high in high_df.groupby("user_id"):
#         gt_item = gt_map[u]
#         gt_time = gt_time_map[u]

#         # 行为序列：所有评分>=4 且 时间 < ground_truth 时间 的 item（避免与 gt 同时刻重复）
#         # 同时也允许时间 <= gt_time 但 item_id != gt_item；更稳妥是直接“去掉 gt 那一条记录”
#         # 这里采用“去掉 item==gt_item 且 timestamp==gt_time 的那条”，剩余按时间升序
#         uh = u_high.copy()
#         uh = uh[~((uh["item_id"] == gt_item) & (uh["timestamp"] == gt_time))]
#         uh = uh.sort_values("timestamp")
#         beh_seq = uh["item_id"].tolist()

#         # 剔除：没有其它 4/5（即 behavior 为空）或本用户只有一条 4/5
#         if len(beh_seq) == 0:
#             continue

#         # 生成 candidate：ground_truth + 19 个未交互
#         user_interacted = interacted.get(u, set())
#         not_interacted = list(all_items - user_interacted)

#         # 可能极端情况下未交互数 < 19（在 ML-100K 基本不会发生），做兜底：允许有放回采样
#         neg_needed = 19
#         negs = []
#         if len(not_interacted) >= neg_needed:
#             negs = random.sample(not_interacted, neg_needed)
#         else:
#             # 先尽量拿去重样本，然后补齐
#             base = not_interacted[:]
#             while len(base) < neg_needed:
#                 base.append(random.choice(not_interacted))
#             negs = base[:neg_needed]

#         cand = [gt_item] + negs
#         random.shuffle(cand)

#         behavior_rows.append({
#             "user_id": u,
#             "behavior": json.dumps(beh_seq, ensure_ascii=False),
#             "ground_truth": gt_item
#         })
#         candidate_rows.append({
#             "user_id": u,
#             "candidate": json.dumps(cand, ensure_ascii=False),
#             "ground_truth": gt_item
#         })

#     behavior_df = pd.DataFrame(behavior_rows)
#     candidate_df = pd.DataFrame(candidate_rows)
#     return behavior_df, candidate_df

# def make_train(df, all_items):
#     """
#     生成 train.csv
#     每用户一行：
#       - log_seq: 该用户按时间升序的所有交互 item（包含所有评分）
#       - pos: 评分为 5 的 item 列表
#       - neg: 从未交互物品中采样 len(pos) 个（不够则有放回）
#     即便某些用户 pos 为空，也会保留该行（便于你后续筛选或处理）。
#     """
#     df_sorted = df.sort_values(["user_id", "timestamp"])
#     interacted = df_sorted.groupby("user_id")["item_id"].apply(list)
#     interacted_set = df_sorted.groupby("user_id")["item_id"].apply(set)

#     pos_map = df_sorted[df_sorted["rating"] == 5].groupby("user_id")["item_id"].apply(list).to_dict()

#     rows = []
#     for u, log_seq in interacted.items():
#         user_all_set = interacted_set[u]
#         pos_list = pos_map.get(u, [])
#         # 采样 neg：与 pos 等长
#         neg_len = len(pos_list)
#         not_interacted = list(all_items - user_all_set)

#         if neg_len > 0:
#             if len(not_interacted) >= neg_len:
#                 neg_list = random.sample(not_interacted, neg_len)
#             else:
#                 # 不足则有放回
#                 neg_list = []
#                 for _ in range(neg_len):
#                     neg_list.append(random.choice(not_interacted))
#         else:
#             neg_list = []

#         rows.append({
#             "user_id": u,
#             "log_seq": json.dumps(log_seq, ensure_ascii=False),
#             "pos": json.dumps(pos_list, ensure_ascii=False),
#             "neg": json.dumps(neg_list, ensure_ascii=False),
#         })

#     return pd.DataFrame(rows)

# def main():
#     os.makedirs(OUT_DIR, exist_ok=True)
#     df, all_items = read_ml100k(RAW_DIR)

#     behavior_df, candidate_df = make_behavior_and_candidate(df, all_items)
#     train_df = make_train(df, all_items)

#     behavior_path = os.path.join(OUT_DIR, "behavior.csv")
#     candidate_path = os.path.join(OUT_DIR, "candidate.csv")
#     train_path = os.path.join(OUT_DIR, "train.csv")

#     behavior_df.to_csv(behavior_path, index=False)
#     candidate_df.to_csv(candidate_path, index=False)
#     train_df.to_csv(train_path, index=False)

#     print(f"Saved: {behavior_path} ({len(behavior_df)} rows)")
#     print(f"Saved: {candidate_path} ({len(candidate_df)} rows)")
#     print(f"Saved: {train_path} ({len(train_df)} rows)")

# if __name__ == "__main__":
#     main()
