import os
import ast
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model.SASRec import SASRec  # prefer external file if exists


@dataclass
class TrainArgs:
    data_dir: str
    save_dir: str
    train_file: str = "train.csv"
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    maxlen: int = 200
    hidden_units: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout_rate: float = 0.2
    norm_first: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utility
# -----------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Dataset that expands each user row into multiple (history -> target) samples
# -----------------------------

class SASRecTrainDataset(Dataset):
    def __init__(self, csv_path: str, maxlen: int):
        df = pd.read_csv(csv_path)
        self.maxlen = maxlen
        self.samples: List[Tuple[int, List[int], int, int]] = []  # (user_id, history, pos_item, neg_item)

        # For computing item_num later
        self._max_item_id = 0

        for row in df.itertuples(index=False):
            user_id = int(getattr(row, 'user_id'))
            log_seq = ast.literal_eval(getattr(row, 'log_seq'))
            pos_list = ast.literal_eval(getattr(row, 'pos'))
            neg_list = ast.literal_eval(getattr(row, 'neg'))

            # Track max item id
            if log_seq:
                self._max_item_id = max(self._max_item_id, max(log_seq))
            if pos_list:
                self._max_item_id = max(self._max_item_id, max(pos_list))
            if neg_list:
                self._max_item_id = max(self._max_item_id, max(neg_list))

            # Build (history -> target) pairs using the chronological index of each positive item
            index_map = {item: idx for idx, item in enumerate(log_seq)}
            for pos_item in pos_list:
                if pos_item not in index_map:
                    # skip if the pos item is not actually in the log (shouldn't happen, but be safe)
                    continue
                idx = index_map[pos_item]
                history = log_seq[:idx]  # prefix before the pos item occurs
                if len(history) == 0:
                    # no history context → skip
                    continue

                # Choose a neg item from provided list; if empty, random from 1..max_id that is not in history
                if len(neg_list) > 0:
                    neg_item = int(np.random.choice(neg_list))
                else:
                    # fallback random (very rare if input prepared well)
                    neg_item = np.random.randint(1, max(self._max_item_id, 2))
                    while neg_item in history or neg_item == pos_item:
                        neg_item = np.random.randint(1, max(self._max_item_id, 2))

                self.samples.append((user_id, history, int(pos_item), int(neg_item)))

        # If no samples, raise a clear error
        if len(self.samples) == 0:
            raise ValueError("No training samples constructed. Check your train.csv format and contents.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, history, pos_item, neg_item = self.samples[idx]
        # Right-pad to maxlen; keep only the last maxlen tokens
        seq = history[-self.maxlen:]
        pad_len = self.maxlen - len(seq)
        if pad_len > 0:
            seq = [0] * pad_len + seq

        # We will place target at the last position (others = 0) for compatibility with the provided SASRec.forward
        pos_seq = [0] * (self.maxlen - 1) + [pos_item]
        neg_seq = [0] * (self.maxlen - 1) + [neg_item]

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos_seq, dtype=torch.long),
            torch.tensor(neg_seq, dtype=torch.long),
        )

    @property
    def max_item_id(self) -> int:
        return self._max_item_id


def collate_fn(batch):
    user_ids = torch.stack([b[0] for b in batch], dim=0)
    log_seqs = torch.stack([b[1] for b in batch], dim=0)
    pos_seqs = torch.stack([b[2] for b in batch], dim=0)
    neg_seqs = torch.stack([b[3] for b in batch], dim=0)
    return user_ids, log_seqs, pos_seqs, neg_seqs


# -----------------------------
# Training utilities
# -----------------------------

def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BPR loss on masked positions.
    pos_logits, neg_logits: (B, T)
    mask: (B, T) boolean; True where valid target exists
    """
    diff = pos_logits - neg_logits
    diff = diff[mask]
    return -(torch.log(torch.sigmoid(diff) + 1e-12)).mean()


def batch_auc(pos_logits: torch.Tensor, neg_logits: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.no_grad():
        correct = (pos_logits > neg_logits) & mask
        total = mask.sum().item()
        if total == 0:
            return 0.0
        return correct.sum().item() / total


# -----------------------------
# Main train loop
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/ml100k/processed')
    parser.add_argument('--save_dir', type=str, default='../model/saved')
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--maxlen', type=int, default=200)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args_cli = parser.parse_args()

    set_seed(args_cli.seed)

    csv_path = os.path.join(args_cli.data_dir, args_cli.train_file)
    dataset = SASRecTrainDataset(csv_path, maxlen=args_cli.maxlen)
    loader = DataLoader(dataset, batch_size=args_cli.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    user_num = int(pd.read_csv(csv_path)['user_id'].max())
    item_num = int(dataset.max_item_id)

    # Build args namespace expected by SASRec
    class ModelArgs:
        pass
    margs = ModelArgs()
    margs.device = args_cli.device
    margs.norm_first = args_cli.norm_first
    margs.hidden_units = args_cli.hidden_units
    margs.dropout_rate = args_cli.dropout_rate
    margs.num_blocks = args_cli.num_blocks
    margs.num_heads = args_cli.num_heads
    margs.maxlen = args_cli.maxlen

    model = SASRec(user_num=user_num, item_num=item_num, args=margs).to(args_cli.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args_cli.lr, weight_decay=args_cli.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=(args_cli.device.startswith('cuda')))
    with torch.amp.autocast("cuda", enabled=(args_cli.device.startswith('cuda'))):
        ensure_dir(args_cli.save_dir)

    for epoch in range(1, args_cli.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_auc = 0.0
        count = 0

        for batch in loader:
            user_ids, log_seqs, pos_seqs, neg_seqs = batch
            log_seqs = log_seqs.to(args_cli.device)
            pos_seqs = pos_seqs.to(args_cli.device)
            neg_seqs = neg_seqs.to(args_cli.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(args_cli.device.startswith('cuda'))):
                pos_logits, neg_logits = model(user_ids=None, log_seqs=log_seqs, pos_seqs=pos_seqs, neg_seqs=neg_seqs)
                # mask only the last position (where target placed)
                mask = pos_seqs.gt(0)
                loss = bpr_loss(pos_logits, neg_logits, mask)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_auc_val = batch_auc(pos_logits, neg_logits, mask)
            bs = log_seqs.size(0)
            epoch_loss += loss.item() * bs
            epoch_auc += batch_auc_val * bs
            count += bs

        avg_loss = epoch_loss / max(count, 1)
        avg_auc = epoch_auc / max(count, 1)
        print(f"Epoch {epoch} | Loss {avg_loss:.4f} | AUC {avg_auc:.4f}")

    # Save model and config
    model_path = os.path.join(args_cli.save_dir, 'sasrec_ml100k_2.pt')
    torch.save({'state_dict': model.state_dict(),
                'user_num': user_num,
                'item_num': item_num,
                'config': {
                    'hidden_units': args_cli.hidden_units,
                    'num_blocks': args_cli.num_blocks,
                    'num_heads': args_cli.num_heads,
                    'dropout_rate': args_cli.dropout_rate,
                    'maxlen': args_cli.maxlen,
                    'norm_first': args_cli.norm_first,
                }}, model_path)

    with open(os.path.join(args_cli.save_dir, 'train_args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args_cli), f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {model_path}")


if __name__ == '__main__':
    main()
