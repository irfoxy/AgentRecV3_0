import os
import json
import torch
import pandas as pd
import numpy as np

from model.SASRec import SASRec

# 路径配置
DATA_DIR = "../data/ml100k/processed"
BEHAVIOR = os.path.join(DATA_DIR, "behavior.csv")
CANDIDATE = os.path.join(DATA_DIR, "candidate.csv")
CKPT = "../model/saved/sasrec_20250922_185909.pt"  # 换成你要评估的模型路径

MAXLEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_left(seq, maxlen):
    seq = seq[-maxlen:]
    return [0] * (maxlen - len(seq)) + seq


def hit_at_1(rank, gt):
    """Hit@1：Top1 是否等于 ground_truth"""
    return 1.0 if len(rank) > 0 and rank[0] == gt else 0.0


def main():
    # 读取数据
    beh = pd.read_csv(BEHAVIOR)
    cand = pd.read_csv(CANDIDATE)

    df = beh.merge(
        cand[["user_id", "candidate", "ground_truth"]],
        on=["user_id", "ground_truth"],
        how="inner"
    )

    # 加载模型
    ckpt = torch.load(CKPT, map_location="cpu")

    class Args: pass
    args = Args()
    args.hidden_units = ckpt["args"]["hidden_units"]
    args.num_blocks = ckpt["args"]["num_blocks"]
    args.num_heads = ckpt["args"]["num_heads"]
    args.dropout_rate = ckpt["args"]["dropout_rate"]
    args.maxlen = ckpt["args"]["maxlen"]
    args.norm_first = ckpt["args"]["norm_first"]
    args.device = DEVICE

    model = SASRec(
        user_num=ckpt["user_num"],
        item_num=ckpt["item_num"],
        args=args
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    hits = []

    with torch.no_grad():
        for _, r in df.iterrows():
            try:
                behavior = json.loads(r["behavior"])
                candidate = json.loads(r["candidate"])
            except Exception:
                import ast
                behavior = ast.literal_eval(r["behavior"])
                candidate = ast.literal_eval(r["candidate"])

            log_seq = torch.tensor(
                [pad_left(list(map(int, behavior)), MAXLEN)],
                dtype=torch.long, device=DEVICE
            )
            items = torch.tensor(
                [list(map(int, candidate))],
                dtype=torch.long, device=DEVICE
            )
            users = torch.tensor(
                [int(r["user_id"])],
                dtype=torch.long, device=DEVICE
            )

            scores = model.predict(users, log_seq, items)  # (1, I)
            scores = scores.squeeze(0).cpu().numpy()
            rank = [x for _, x in sorted(zip(scores, candidate),
                                         key=lambda t: t[0], reverse=True)]

            hit = hit_at_1(rank, int(r["ground_truth"]))
            hits.append(hit)

    print(f"Hit@1: {np.mean(hits):.4f}  (Users={len(hits)})")


if __name__ == "__main__":
    main()
