#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset.py — 从 ../processed/behavior.csv 生成 ./candidate.csv

规则：
- 对于每个用户（behavior.csv 的一行）：
  - 取 behavior 序列的最后一个 item_id 作为正样本 (ground_truth)；
  - 从 [1..1682] 中随机选择 19 个不在该用户 behavior 中出现的 item_id 作为负样本；
  - 输出一行：user_id, candidate, ground_truth
    - candidate: 20 个 item_id，以空格分隔，第一个为 ground_truth，其余 19 个为负样本
    - ground_truth: 即最后一个正样本 item_id
"""

import os
import sys
import random
import pandas as pd
from typing import List, Set

# 固定路径（相对脚本位置）
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IN_DIR = os.path.abspath(os.path.join(BASE_DIR, "../processed"))
OUT_DIR = BASE_DIR

# 常量
N_ITEMS = 1682               # MovieLens 100k 物品总数
RNG_SEED = 2025              # 固定随机种子，保证可复现；如需真正随机可改为 None

def _err(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def _parse_behavior(behavior_str: str) -> List[int]:
    """将空格分隔的 item_id 串解析为 int 列表；去除空白和非数字噪声。"""
    if not isinstance(behavior_str, str) or not behavior_str.strip():
        return []
    out = []
    for tok in behavior_str.strip().split():
        try:
            out.append(int(tok))
        except Exception:
            continue
    return out

def main():
    # 输入文件
    beh_path = os.path.join(IN_DIR, "behavior.csv")
    if not os.path.exists(beh_path):
        _err(f"Missing file: {beh_path}")

    df = pd.read_csv(beh_path)

    # 检查字段
    if "user_id" not in df.columns or "behavior" not in df.columns:
        _err("behavior.csv must contain columns: user_id, behavior")

    rng = random.Random(RNG_SEED)
    universe: Set[int] = set(range(1, N_ITEMS + 1))

    out_rows = []
    for _, row in df.iterrows():
        user_id = row["user_id"]
        seq = _parse_behavior(row["behavior"])

        if not seq:
            continue

        # ground truth = 最后一个交互
        ground_truth = int(seq[-1])

        # 负采样池
        seen = set(int(i) for i in seq if 1 <= int(i) <= N_ITEMS)
        neg_pool = list(universe - seen)

        if len(neg_pool) < 19:
            _err(f"Not enough negatives to sample for user_id={user_id}")

        negatives = rng.sample(neg_pool, 19)

        # 组合 candidate（正样本在前）
        candidate_items = [ground_truth] + negatives
        candidate_str = " ".join(map(str, candidate_items))

        out_rows.append({
            "user_id": user_id,
            "candidate": candidate_str,
            "ground_truth": ground_truth
        })

    out_df = pd.DataFrame(out_rows, columns=["user_id", "candidate", "ground_truth"])
    out_path = os.path.join(OUT_DIR, "candidate.csv")
    out_df.to_csv(out_path, index=False)

    print("Done. Wrote ./candidate.csv")

if __name__ == "__main__":
    main()
