# import os
# import json
# import time
# import math
# import random
# import argparse
# from typing import List

# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader

# from model.SASRec import SASRec


# # ============ 数据集 ============

# class TrainDataset(Dataset):
#     def __init__(self, csv_path: str, maxlen: int):
#         self.maxlen = maxlen
#         df = pd.read_csv(csv_path)

#         rows = []
#         for _, r in df.iterrows():
#             try:
#                 log_seq = json.loads(r["log_seq"])
#                 pos = json.loads(r["pos"])
#                 neg = json.loads(r["neg"])
#             except Exception:
#                 import ast
#                 log_seq = ast.literal_eval(r["log_seq"])
#                 pos = ast.literal_eval(r["pos"])
#                 neg = ast.literal_eval(r["neg"])

#             # 跳过无正/负样本的行
#             if not pos or not neg:
#                 continue

#             # 负样本裁剪/补齐到与正样本等长
#             if len(neg) > len(pos):
#                 neg = neg[:len(pos)]
#             while len(neg) < len(pos):
#                 neg.append(random.choice(neg))

#             rows.append((
#                 int(r["user_id"]),
#                 list(map(int, log_seq)),
#                 list(map(int, pos)),
#                 list(map(int, neg)),
#             ))

#         self.rows = rows

#     def __len__(self):
#         return len(self.rows)

#     def _pad_left(self, seq: List[int]) -> List[int]:
#         seq = seq[-self.maxlen:]
#         return [0] * (self.maxlen - len(seq)) + seq

#     def __getitem__(self, idx):
#         user, log_seq, pos, neg = self.rows[idx]

#         pos = pos[-self.maxlen:]
#         neg = neg[-self.maxlen:]
#         pos = [0] * (self.maxlen - len(pos)) + pos
#         neg = [0] * (self.maxlen - len(neg)) + neg

#         log_seq = self._pad_left(log_seq)

#         return (
#             torch.tensor(user, dtype=torch.long),
#             torch.tensor(log_seq, dtype=torch.long),
#             torch.tensor(pos, dtype=torch.long),
#             torch.tensor(neg, dtype=torch.long),
#         )


# # ============ 训练工具函数 ============

# def bpr_loss_masked(pos_logits, neg_logits, mask):
#     """
#     pos_logits, neg_logits: (B, T)
#     mask: (B, T) in {0,1}
#     只在 mask==1 的位置计算损失
#     """
#     eps = 1e-24
#     diff = pos_logits - neg_logits
#     loss = -torch.log(torch.sigmoid(diff) + eps) * mask
#     denom = mask.sum().clamp_min(1.0)
#     return loss.sum() / denom

# @torch.no_grad()
# def pairwise_auc_masked(pos_logits, neg_logits, mask):
#     """
#     AUC: 有效位置上 pos > neg 的比例
#     """
#     correct = ((pos_logits > neg_logits).float() * mask).sum()
#     denom = mask.sum().clamp_min(1.0)
#     return (correct / denom).item()


# def infer_user_item_num(csv_path: str):
#     """从 train.csv 估计 user_num 与 item_num（最大 id）。"""
#     df = pd.read_csv(csv_path)
#     max_user = int(df["user_id"].max())

#     def max_from_col(col):
#         m = 0
#         for s in df[col].tolist():
#             try:
#                 arr = json.loads(s)
#             except Exception:
#                 import ast
#                 arr = ast.literal_eval(s)
#             m = max(m, max(arr) if len(arr) else 0)
#         return m

#     max_item = max(max_from_col("log_seq"), max_from_col("pos"), max_from_col("neg"))
#     return max_user, max_item


# # ============ 主训练流程 ============

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=str, default="../data/ml100k/processed")
#     parser.add_argument("--train_file", type=str, default="train.csv")
#     parser.add_argument("--save_dir", type=str, default="../model/saved")

#     # 模型与训练超参
#     parser.add_argument("--hidden_units", type=int, default=128)
#     parser.add_argument("--num_blocks", type=int, default=2)
#     parser.add_argument("--num_heads", type=int, default=2)
#     parser.add_argument("--dropout_rate", type=float, default=0.2)
#     parser.add_argument("--maxlen", type=int, default=50)
#     parser.add_argument("--norm_first", action="store_true", help="使用 pre-norm transformer")
#     parser.add_argument("--batch_size", type=int, default=128)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--weight_decay", type=float, default=0.0, help=">0 时使用 AdamW L2 正则")
#     parser.add_argument("--use_scheduler", action="store_true", help="启用 CosineAnnealingLR 调度")

#     # 原有：只最后一步的 pairwise-BPR（保留兼容）
#     parser.add_argument("--last_step_only", action="store_true",
#                         help="只在最后一步用 pairwise-BPR 监督（与 listwise_ce 互斥）")

#     # 方案1：候选集交叉熵（listwise）
#     parser.add_argument("--listwise_ce", action="store_true",
#                         help="用候选集交叉熵训练（最后一步 1 正 + K 负，默认 K=19）")
#     parser.add_argument("--negs_per_pos", type=int, default=19,
#                         help="每个样本的负样本个数（与验证20候选对齐）")
#     parser.add_argument("--tau", type=float, default=1.0,
#                         help="softmax 温度缩放（<1 稍微拉大分差）")
#     parser.add_argument("--margin", type=float, default=0.0,
#                         help="正样本打分的加性 margin（>0 拉开正负间隔）")

#     parser.add_argument("--seed", type=int, default=2024)

#     args = parser.parse_args()

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     args.device = device

#     # 路径
#     train_path = os.path.join(args.data_dir, args.train_file)
#     os.makedirs(args.save_dir, exist_ok=True)

#     # 推断 user_num / item_num
#     user_num, item_num = infer_user_item_num(train_path)
#     print(f"[Info] user_num={user_num}, item_num={item_num}")

#     # 数据
#     dataset = TrainDataset(train_path, maxlen=args.maxlen)
#     if len(dataset) == 0:
#         raise RuntimeError("训练数据为空：请检查 train.csv 是否存在有效的 pos/neg。")
#     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

#     # 构建模型
#     model = SASRec(user_num=user_num, item_num=item_num, args=args).to(device)

#     if args.weight_decay > 0:
#         optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     else:
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     # 学习率调度（可选）
#     if args.use_scheduler:
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1
#         )
#     else:
#         scheduler = None

#     # listwise 与 last_step_only 同时开启会冲突，这里优先 listwise
#     if args.listwise_ce and args.last_step_only:
#         print("[Warn] --listwise_ce 与 --last_step_only 同时开启，已忽略 --last_step_only。")
#         args.last_step_only = False

#     # 训练
#     print(f"[Start] epochs={args.epochs}, batches/epoch={math.ceil(len(dataset)/args.batch_size)}")
#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         epoch_loss, epoch_metric, n_batches = 0.0, 0.0, 0

#         for batch in loader:
#             user_ids, log_seqs, pos_seqs, neg_seqs = [x.to(device) for x in batch]

#             if args.listwise_ce:
#                 # ====== 仅最后一步监督：1 正 + K 负 做 softmax-CE ======
#                 final_pos = pos_seqs[:, -1]                  # (B,)
#                 mask_last = (final_pos != 0)                 # 有效样本
#                 if mask_last.sum() == 0:
#                     continue

#                 B = final_pos.size(0)
#                 K = args.negs_per_pos

#                 # in-batch negatives（更难）：从别的样本的正样本构造，尽量避免与自身相同
#                 neg_cols = []
#                 needed = min(max(B - 1, 1), K)
#                 for _ in range(needed):
#                     shuf = final_pos[torch.randperm(B)]
#                     # 避免与自身正样本相同（再洗一次）
#                     same = (shuf == final_pos)
#                     if same.any():
#                         shuf2 = final_pos[torch.randperm(B)]
#                         shuf = torch.where(same, shuf2, shuf)
#                     neg_cols.append(shuf)
#                 if len(neg_cols) > 0:
#                     inbatch_negs = torch.stack(neg_cols, dim=1)  # (B, needed)
#                 else:
#                     inbatch_negs = torch.empty(B, 0, device=device, dtype=final_pos.dtype)

#                 # 若不足 K，随机补齐
#                 remain = K - inbatch_negs.size(1)
#                 if remain > 0:
#                     rand_negs = torch.randint(low=1, high=item_num + 1, size=(B, remain), device=device)
#                     neg_matrix = torch.cat([inbatch_negs, rand_negs], dim=1)
#                 else:
#                     neg_matrix = inbatch_negs[:, :K]

#                 # 组装候选 (B, 1+K): [pos | negs]
#                 items = torch.cat([final_pos.view(-1, 1), neg_matrix], dim=1)

#                 # 对 (B, 1+K) 候选打分（SASRec.predict 用最后一步特征）
#                 scores = model.predict(user_ids, log_seqs, items)  # (B, 1+K)

#                 # 可选：温度缩放 & 正样本 margin（对 index 0）
#                 if args.margin != 0.0:
#                     scores[:, 0] = scores[:, 0] - args.margin
#                 if args.tau != 1.0:
#                     scores = scores / args.tau

#                 labels = torch.zeros(scores.size(0), dtype=torch.long, device=device)  # 正样本在 0
#                 scores_valid = scores[mask_last]
#                 labels_valid = labels[mask_last]

#                 loss = torch.nn.functional.cross_entropy(scores_valid, labels_valid)

#                 # 在线指标：Hit@1（与 valid 一致）
#                 with torch.no_grad():
#                     hit1 = (scores_valid.argmax(dim=1) == labels_valid).float().mean().item()
#                 batch_metric = hit1

#             else:
#                 # ====== 保留原来的 masked-BPR 训练分支 ======
#                 pos_logits, neg_logits = model(user_ids, log_seqs, pos_seqs, neg_seqs)

#                 if args.last_step_only:
#                     # 仅最后一步监督的 BPR
#                     pos_last = pos_logits[:, -1]
#                     neg_last = neg_logits[:, -1]
#                     mask = (pos_seqs[:, -1] != 0).float()
#                     eps = 1e-24
#                     diff = pos_last - neg_last
#                     loss = -torch.log(torch.sigmoid(diff) + eps) * mask
#                     denom = mask.sum().clamp_min(1.0)
#                     loss = loss.sum() / denom
#                     with torch.no_grad():
#                         auc = ((pos_last > neg_last).float() * mask).sum() / denom
#                         auc = auc.item()
#                     batch_metric = auc
#                 else:
#                     # 全序列监督 + padding mask（只在 pos!=0 的位置）
#                     mask = (pos_seqs != 0).float()  # (B, T)
#                     loss = bpr_loss_masked(pos_logits, neg_logits, mask)
#                     with torch.no_grad():
#                         batch_metric = pairwise_auc_masked(pos_logits, neg_logits, mask)

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()

#             epoch_loss += loss.item()
#             epoch_metric += batch_metric
#             n_batches += 1

#             # print(f"Epoch {epoch} | Batch {n_batches} | "
#             #       f"{'CE' if args.listwise_ce else ('BPR-last' if args.last_step_only else 'BPR')} {loss.item():.4f} | "
#             #       f"{'Hit@1' if args.listwise_ce else 'AUC'} {batch_metric:.4f}")

#         print(f"===> Epoch {epoch} Finished | AvgLoss {epoch_loss/n_batches:.4f} | "
#               f"{'AvgHit@1' if args.listwise_ce else 'AvgAUC'} {epoch_metric/n_batches:.4f}")

#         if scheduler is not None:
#             scheduler.step()

#     # 保存
#     ts = time.strftime("%Y%m%d_%H%M%S")
#     save_path = os.path.join(args.save_dir, f"sasrec_{ts}.pt")
#     torch.save({
#         "model_state": model.state_dict(),
#         "user_num": user_num,
#         "item_num": item_num,
#         "args": {
#             "hidden_units": args.hidden_units,
#             "num_blocks": args.num_blocks,
#             "num_heads": args.num_heads,
#             "dropout_rate": args.dropout_rate,
#             "maxlen": args.maxlen,
#             "norm_first": args.norm_first,
#         }
#     }, save_path)
#     print(f"[Saved] {save_path}")


# if __name__ == "__main__":
#     main()

import os
import json
import time
import math
import random
import argparse
from typing import List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model.SASRec import SASRec


# ============ 数据集 ============

class TrainDataset(Dataset):
    def __init__(self, csv_path: str, maxlen: int):
        self.maxlen = maxlen
        df = pd.read_csv(csv_path)

        rows = []
        for _, r in df.iterrows():
            try:
                log_seq = json.loads(r["log_seq"])
                pos = json.loads(r["pos"])
                neg = json.loads(r["neg"])
            except Exception:
                import ast
                log_seq = ast.literal_eval(r["log_seq"])
                pos = ast.literal_eval(r["pos"])
                neg = ast.literal_eval(r["neg"])

            # 跳过无正/负样本的行
            if not pos or not neg:
                continue

            # 负样本裁剪/补齐到与正样本等长
            if len(neg) > len(pos):
                neg = neg[:len(pos)]
            while len(neg) < len(pos):
                neg.append(random.choice(neg))

            rows.append((
                int(r["user_id"]),
                list(map(int, log_seq)),
                list(map(int, pos)),
                list(map(int, neg)),
            ))

        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _pad_left(self, seq: List[int]) -> List[int]:
        seq = seq[-self.maxlen:]
        return [0] * (self.maxlen - len(seq)) + seq

    def __getitem__(self, idx):
        user, log_seq, pos, neg = self.rows[idx]

        pos = pos[-self.maxlen:]
        neg = neg[-self.maxlen:]
        pos = [0] * (self.maxlen - len(pos)) + pos
        neg = [0] * (self.maxlen - len(neg)) + neg

        log_seq = self._pad_left(log_seq)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(log_seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


# ============ 训练工具函数 ============

def bpr_loss_masked(pos_logits, neg_logits, mask):
    """
    pos_logits, neg_logits: (B, T)
    mask: (B, T) in {0,1}
    只在 mask==1 的位置计算损失
    """
    eps = 1e-24
    diff = pos_logits - neg_logits
    loss = -torch.log(torch.sigmoid(diff) + eps) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom

@torch.no_grad()
def pairwise_auc_masked(pos_logits, neg_logits, mask):
    """
    AUC: 有效位置上 pos > neg 的比例
    """
    correct = ((pos_logits > neg_logits).float() * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return (correct / denom).item()


def infer_user_item_num(csv_path: str):
    """从 train.csv 估计 user_num 与 item_num（最大 id）。"""
    df = pd.read_csv(csv_path)
    max_user = int(df["user_id"].max())

    def max_from_col(col):
        m = 0
        for s in df[col].tolist():
            try:
                arr = json.loads(s)
            except Exception:
                import ast
                arr = ast.literal_eval(s)
            m = max(m, max(arr) if len(arr) else 0)
        return m

    max_item = max(max_from_col("log_seq"), max_from_col("pos"), max_from_col("neg"))
    return max_user, max_item


# ============ 主训练流程 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/ml100k/processed")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--save_dir", type=str, default="../model/saved")

    # 模型与训练超参
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--norm_first", action="store_true", help="使用 pre-norm transformer")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0, help=">0 时使用 AdamW L2 正则")
    parser.add_argument("--use_scheduler", action="store_true", help="启用 CosineAnnealingLR 调度")
    parser.add_argument("--last_step_only", action="store_true", help="只在最后一步监督（final_feat 与 pos/neg 的最后一位对比）")
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 路径
    train_path = os.path.join(args.data_dir, args.train_file)
    os.makedirs(args.save_dir, exist_ok=True)

    # 推断 user_num / item_num
    user_num, item_num = infer_user_item_num(train_path)
    print(f"[Info] user_num={user_num}, item_num={item_num}")

    # 数据
    dataset = TrainDataset(train_path, maxlen=args.maxlen)
    if len(dataset) == 0:
        raise RuntimeError("训练数据为空：请检查 train.csv 是否存在有效的 pos/neg。")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # 构建模型
    model = SASRec(user_num=user_num, item_num=item_num, args=args).to(device)

    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度（可选）
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1
        )
    else:
        scheduler = None

    # 训练
    print(f"[Start] epochs={args.epochs}, batches/epoch={math.ceil(len(dataset)/args.batch_size)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_auc, n_batches = 0.0, 0.0, 0

        for batch in loader:
            user_ids, log_seqs, pos_seqs, neg_seqs = [x.to(device) for x in batch]

            # 前向
            pos_logits, neg_logits = model(user_ids, log_seqs, pos_seqs, neg_seqs)

            if args.last_step_only:
                # 仅最后一步监督：取最后一位 (B,)
                pos_last = pos_logits[:, -1]
                neg_last = neg_logits[:, -1]
                # mask：最后一位是否有效（pos!=0）
                mask = (pos_seqs[:, -1] != 0).float()
                # 扩展到 (B,) 的 BPR
                eps = 1e-24
                diff = pos_last - neg_last
                loss = -torch.log(torch.sigmoid(diff) + eps) * mask
                denom = mask.sum().clamp_min(1.0)
                loss = loss.sum() / denom
                with torch.no_grad():
                    auc = ((pos_last > neg_last).float() * mask).sum() / denom
                    auc = auc.item()
            else:
                # 全序列监督 + padding mask（只在 pos!=0 的位置）
                mask = (pos_seqs != 0).float()  # (B, T)
                loss = bpr_loss_masked(pos_logits, neg_logits, mask)
                with torch.no_grad():
                    auc = pairwise_auc_masked(pos_logits, neg_logits, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_auc += auc
            n_batches += 1

            # print(f"Epoch {epoch} | Batch {n_batches} | Loss {loss.item():.4f} | AUC {auc:.4f}")

        print(f"===> Epoch {epoch} Finished | AvgLoss {epoch_loss/n_batches:.4f} | AvgAUC {epoch_auc/n_batches:.4f}")

        if scheduler is not None:
            scheduler.step()

    # 保存
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f"sasrec_{ts}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "user_num": user_num,
        "item_num": item_num,
        "args": {
            "hidden_units": args.hidden_units,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
            "dropout_rate": args.dropout_rate,
            "maxlen": args.maxlen,
            "norm_first": args.norm_first,
        }
    }, save_path)
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
