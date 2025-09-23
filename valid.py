import os
import ast
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Prefer external SASRec definition if provided
from model.SASRec import SASRec


def load_train_args(save_dir: str) -> Dict:
    cfg_path = os.path.join(save_dir, 'train_args.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_model_from_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    class ModelArgs:
        pass

    margs = ModelArgs()
    model_cfg = ckpt.get('config', {})
    margs.device = device
    margs.norm_first = model_cfg.get('norm_first', True)
    margs.hidden_units = model_cfg.get('hidden_units', 64)
    margs.dropout_rate = model_cfg.get('dropout_rate', 0.2)
    margs.num_blocks = model_cfg.get('num_blocks', 2)
    margs.num_heads = model_cfg.get('num_heads', 2)
    margs.maxlen = model_cfg.get('maxlen', 200)

    model = SASRec(user_num=ckpt['user_num'], item_num=ckpt['item_num'], args=margs).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model, margs


def right_pad(seq: List[int], maxlen: int) -> List[int]:
    seq = seq[-maxlen:]
    pad = maxlen - len(seq)
    if pad > 0:
        seq = [0] * pad + seq
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/ml100k/processed')
    parser.add_argument('--save_dir', type=str, default='../model/saved')
    parser.add_argument('--candidate_file', type=str, default='candidate.csv')
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--print_limit', type=int, default=5, help='仅打印前N行候选分数')
    args = parser.parse_args()

    device = args.device
    ckpt_path = os.path.join(args.save_dir, 'sasrec_ml100k.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    model, margs = build_model_from_ckpt(ckpt_path, device)

    # 读取用户历史
    train_csv = os.path.join(args.data_dir, args.train_file)
    df_train = pd.read_csv(train_csv)
    user2hist: Dict[int, List[int]] = {}
    for row in df_train.itertuples(index=False):
        uid = int(getattr(row, 'user_id'))
        log_seq = ast.literal_eval(getattr(row, 'log_seq'))
        user2hist[uid] = log_seq

    # 读取候选集
    cand_csv = os.path.join(args.data_dir, args.candidate_file)
    df_cand = pd.read_csv(cand_csv)

    hits = []
    with torch.no_grad():
        for ridx, row in enumerate(df_cand.itertuples(index=False)):
            uid = int(getattr(row, 'user_id'))
            candidate = ast.literal_eval(getattr(row, 'candidate'))
            gt = int(getattr(row, 'ground_truth'))

            log_seq = user2hist.get(uid, [])

            log_tensor = torch.tensor([right_pad(log_seq, margs.maxlen)], dtype=torch.long, device=device)
            cand_tensor = torch.tensor([candidate], dtype=torch.long, device=device)

            scores = model.predict(user_ids=None, log_seqs=log_tensor, item_indices=cand_tensor)  # (1, I)
            scores1 = scores.squeeze(0)  # (I,)
            top1_idx = int(torch.argmax(scores1).item())
            pred_item = candidate[top1_idx]
            hit = 1 if pred_item == gt else 0
            hits.append(hit)

            # —— 仅打印前 N 行的详细分数 —— #
            if ridx < args.print_limit:
                scores_list = scores1.detach().cpu().tolist()
                # 计算 GT 的下标与排名（1-based）
                gt_idx = candidate.index(gt) if gt in candidate else -1
                order = torch.argsort(scores1, descending=True).tolist()
                gt_rank = (order.index(gt_idx) + 1) if gt_idx >= 0 else None

                print(f"\nRow {ridx} | user {uid} | GT={gt} | GT_idx={gt_idx} | GT_rank={gt_rank}")
                for i, (it, sc) in enumerate(zip(candidate, scores_list)):
                    mark = " <== GT" if it == gt else ""
                    print(f"  cand[{i:02d}] item={it:>5}  score={sc:.6f}{mark}")

    hit1 = float(np.mean(hits)) if hits else 0.0
    print(f"\nHit@1: {hit1:.4f}  (N={len(hits)})")


if __name__ == '__main__':
    main()
