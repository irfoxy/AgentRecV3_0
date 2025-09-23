#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
from typing import List, Set

import pandas as pd

# 固定路径
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IN_DIR = os.path.abspath(os.path.join(BASE_DIR, "../processed"))
OUT_DIR = BASE_DIR

# 常量
N_ITEMS = 1682        # MovieLens 100k 物品总数
NEG_RATIO = 1.0       # 负样本与正样本的比例 (1.0 表示数量相同)
RNG_SEED = 2025       # 固定随机种子以便复现 (如需真正随机可设为 None)

def _err(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def _parse_behavior(s: str) -> List[int]:
    """把空格分隔的行为串解析为 int 列表。"""
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for tok in s.strip().split():
        try:
            v = int(tok)
            if 1 <= v <= N_ITEMS:
                out.append(v)
        except Exception:
            continue
    return out

def main():
    beh_path = os.path.join(IN_DIR, "behavior.csv")
    if not os.path.exists(beh_path):
        _err(f"Missing file: {beh_path}")

    df = pd.read_csv(beh_path)
    if "user_id" not in df.columns or "behavior" not in df.columns:
        _err("behavior.csv must contain columns: user_id, behavior")

    rng = random.Random(RNG_SEED)
    universe: Set[int] = set(range(1, N_ITEMS + 1))

    rows = []
    for _, row in df.iterrows():
        user_id = row["user_id"]
        seq = _parse_behavior(row["behavior"])

        # 若该用户没有行为，跳过
        if not seq:
            continue

        # 正样本集合（去重）
        pos_set = list(sorted(set(seq)))
        k_pos = len(pos_set)

        # 负样本数量
        k_neg = max(1, round(k_pos * NEG_RATIO))

        # 负采样池 = 未交互的物品
        seen = set(pos_set)
        neg_pool = list(universe - seen)
        if len(neg_pool) < k_neg:
            _err(f"Not enough negatives for user {user_id}: need {k_neg}, pool {len(neg_pool)}")

        negatives = rng.sample(neg_pool, k_neg)

        # 组合并打乱（保持 sample 与 label 对齐）
        sample = pos_set + negatives
        label = [1] * k_pos + [0] * k_neg

        idx = list(range(len(sample)))
        rng.shuffle(idx)
        sample = [sample[i] for i in idx]
        label = [label[i] for i in idx]

        rows.append({
            "user_id": user_id,
            "sample": json.dumps(sample, ensure_ascii=False),
            "label": json.dumps(label, ensure_ascii=False),
        })

    out_df = pd.DataFrame(rows, columns=["user_id", "sample", "label"])
    out_path = os.path.join(OUT_DIR, "train.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Done. Wrote {out_path}")

if __name__ == "__main__":
    main()
