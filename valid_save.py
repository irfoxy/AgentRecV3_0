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
    parser.add_argument('--print_limit', type=int, default=0, help='仅打印前N行候选分数')
    parser.add_argument('--topk', type=int, default=5, help='导出predicted的Top-K个物品')
    args = parser.parse_args()

    device = args.device
    ckpt_path = os.path.join(args.save_dir, 'sasrec_ml100k_2.pt')
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

    # 命中统计
    hits1, hits3, hits5 = [], [], []
    export_rows = []  # 写 predicted.csv 的中间缓存

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

            # ===== Hit@1 / @3 / @5 =====
            # 找到 GT 在候选中的索引
            gt_idx = candidate.index(gt) if gt in candidate else -1
            if gt_idx == -1:
                # 候选中没有GT，三种命中均为0
                hits1.append(0); hits3.append(0); hits5.append(0)
            else:
                I = scores1.numel()
                order_idx = torch.topk(scores1, k=min(5, I), largest=True, sorted=True).indices.tolist()
                # 命中判定
                hits1.append(1 if gt_idx == order_idx[0] else 0)
                hits3.append(1 if gt_idx in order_idx[:min(3, I)] else 0)
                hits5.append(1 if gt_idx in order_idx[:min(5, I)] else 0)

            # ===== 导出 Top-K 预测（按分数从高到低），存为 Python 列表字符串 =====
            k_pred = min(args.topk, scores1.numel())
            topk_idx = torch.topk(scores1, k=k_pred, largest=True, sorted=True).indices.tolist()
            predicted_items = [candidate[i] for i in topk_idx]
            export_rows.append({
                'user_id': uid,
                'predicted': str(predicted_items)  # 例如 "[256, 895, 949]"
            })

            # —— 仅打印前 N 行的详细分数 —— #
            if ridx < args.print_limit:
                scores_list = scores1.detach().cpu().tolist()
                order_for_print = torch.argsort(scores1, descending=True).tolist()
                gt_rank = (order_for_print.index(gt_idx) + 1) if (gt_idx != -1 and gt_idx in order_for_print) else None

                print(f"\nRow {ridx} | user {uid} | GT={gt} | GT_idx={gt_idx} | GT_rank={gt_rank}")
                for i, (it, sc) in enumerate(zip(candidate, scores_list)):
                    mark = " <== GT" if it == gt else ""
                    print(f"  cand[{i:02d}] item={it:>5}  score={sc:.6f}{mark}")

    # 写出 predicted.csv（路径随 --data_dir 变化）
    out_csv = os.path.join(args.data_dir, 'predicted.csv')
    pd.DataFrame(export_rows).to_csv(out_csv, index=False, encoding='utf-8')
    print(f"\nSaved Top-{args.topk} predictions to: {out_csv}")

    # ===== 汇总指标 =====
    def avg(x): return float(np.mean(x)) if len(x) > 0 else 0.0
    hit1, hit3, hit5 = avg(hits1), avg(hits3), avg(hits5)
    N = len(hits1)
    print(f"Hit@1: {hit1:.4f} | Hit@3: {hit3:.4f} | Hit@5: {hit5:.4f}  (N={N})")


if __name__ == '__main__':
    main()
