import os
import json
import random
import pandas as pd

RAW_DIR = "../raw"   # 输入目录
OUT_DIR = "./"       # 输出目录
SEED = 2024
random.seed(SEED)

def read_ml100k(raw_dir):
    # u.data: user_id, item_id, rating, timestamp (tab 分隔)
    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(
        os.path.join(raw_dir, "u.data"),
        sep="\t",
        names=cols,
        engine="python"
    )
    # u.item: 取第一列的 movie_id 作为全集 item 集合
    # 注意：u.item 用 | 分隔，第一列为电影ID
    item_df = pd.read_csv(
        os.path.join(raw_dir, "u.item"),
        sep="|",
        header=None,
        encoding="latin-1",
        engine="python"
    )
    item_df = item_df.rename(columns={0: "item_id"})
    all_items = set(item_df["item_id"].tolist())
    return df, all_items

def make_behavior_and_candidate(df, all_items):
    """
    生成 behavior.csv 与 candidate.csv
    规则：
      - ground_truth: 用户最近一次评分为 4/5 的 item
      - behavior: 该用户除 ground_truth 外其它评分为 4/5 的 item，按时间升序
      - 若用户没有 4/5 或只有 ground_truth 而无其它 4/5，则剔除该用户
      - candidate: 对每个用户构造 ground_truth + 19 个未交互 item（打乱）
    """
    # 仅保留评分 >=4 的记录做高分筛选
    high_df = df[df["rating"] >= 4].copy()
    if high_df.empty:
        return pd.DataFrame(columns=["user_id","behavior","ground_truth"]), pd.DataFrame(columns=["user_id","candidate","ground_truth"])

    # 为了找最近一次 4/5，按 timestamp 升序后再 groupby 取最后一个
    high_df = high_df.sort_values(["user_id", "timestamp"])
    # 最近一次 4/5 的行作为 ground_truth
    last_high = high_df.groupby("user_id").tail(1)
    gt_map = dict(zip(last_high["user_id"], last_high["item_id"]))
    gt_time_map = dict(zip(last_high["user_id"], last_high["timestamp"]))

    # 构造 behavior：该用户其它 4/5（去掉 ground_truth），按时间升序
    behavior_rows = []
    candidate_rows = []

    # 用户全交互集合（任何评分）用于确定“未交互”
    interacted = df.groupby("user_id")["item_id"].apply(set).to_dict()

    for u, u_high in high_df.groupby("user_id"):
        gt_item = gt_map[u]
        gt_time = gt_time_map[u]

        # 行为序列：所有评分>=4 且 时间 < ground_truth 时间 的 item（避免与 gt 同时刻重复）
        # 同时也允许时间 <= gt_time 但 item_id != gt_item；更稳妥是直接“去掉 gt 那一条记录”
        # 这里采用“去掉 item==gt_item 且 timestamp==gt_time 的那条”，剩余按时间升序
        uh = u_high.copy()
        uh = uh[~((uh["item_id"] == gt_item) & (uh["timestamp"] == gt_time))]
        uh = uh.sort_values("timestamp")
        beh_seq = uh["item_id"].tolist()

        # 剔除：没有其它 4/5（即 behavior 为空）或本用户只有一条 4/5
        if len(beh_seq) == 0:
            continue

        # 生成 candidate：ground_truth + 19 个未交互
        user_interacted = interacted.get(u, set())
        not_interacted = list(all_items - user_interacted)

        # 可能极端情况下未交互数 < 19（在 ML-100K 基本不会发生），做兜底：允许有放回采样
        neg_needed = 19
        negs = []
        if len(not_interacted) >= neg_needed:
            negs = random.sample(not_interacted, neg_needed)
        else:
            # 先尽量拿去重样本，然后补齐
            base = not_interacted[:]
            while len(base) < neg_needed:
                base.append(random.choice(not_interacted))
            negs = base[:neg_needed]

        cand = [gt_item] + negs
        random.shuffle(cand)

        behavior_rows.append({
            "user_id": u,
            "behavior": json.dumps(beh_seq, ensure_ascii=False),
            "ground_truth": gt_item
        })
        candidate_rows.append({
            "user_id": u,
            "candidate": json.dumps(cand, ensure_ascii=False),
            "ground_truth": gt_item
        })

    behavior_df = pd.DataFrame(behavior_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    return behavior_df, candidate_df

def make_train(df, all_items):
    """
    生成 train.csv
    每用户一行：
      - log_seq: 该用户按时间升序的所有交互 item（包含所有评分）
      - pos: 评分为 5 的 item 列表
      - neg: 从未交互物品中采样 len(pos) 个（不够则有放回）
    即便某些用户 pos 为空，也会保留该行（便于你后续筛选或处理）。
    """
    df_sorted = df.sort_values(["user_id", "timestamp"])
    interacted = df_sorted.groupby("user_id")["item_id"].apply(list)
    interacted_set = df_sorted.groupby("user_id")["item_id"].apply(set)

    pos_map = df_sorted[df_sorted["rating"] == 5].groupby("user_id")["item_id"].apply(list).to_dict()

    rows = []
    for u, log_seq in interacted.items():
        user_all_set = interacted_set[u]
        pos_list = pos_map.get(u, [])
        # 采样 neg：与 pos 等长
        neg_len = len(pos_list)
        not_interacted = list(all_items - user_all_set)

        if neg_len > 0:
            if len(not_interacted) >= neg_len:
                neg_list = random.sample(not_interacted, neg_len)
            else:
                # 不足则有放回
                neg_list = []
                for _ in range(neg_len):
                    neg_list.append(random.choice(not_interacted))
        else:
            neg_list = []

        rows.append({
            "user_id": u,
            "log_seq": json.dumps(log_seq, ensure_ascii=False),
            "pos": json.dumps(pos_list, ensure_ascii=False),
            "neg": json.dumps(neg_list, ensure_ascii=False),
        })

    return pd.DataFrame(rows)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, all_items = read_ml100k(RAW_DIR)

    behavior_df, candidate_df = make_behavior_and_candidate(df, all_items)
    train_df = make_train(df, all_items)

    behavior_path = os.path.join(OUT_DIR, "behavior.csv")
    candidate_path = os.path.join(OUT_DIR, "candidate.csv")
    train_path = os.path.join(OUT_DIR, "train.csv")

    behavior_df.to_csv(behavior_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    train_df.to_csv(train_path, index=False)

    print(f"Saved: {behavior_path} ({len(behavior_df)} rows)")
    print(f"Saved: {candidate_path} ({len(candidate_df)} rows)")
    print(f"Saved: {train_path} ({len(train_df)} rows)")

if __name__ == "__main__":
    main()
