import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        # 统一设备：以 embedding 权重所在设备为准
        device = self.item_emb.weight.device

        # 保证输入是 LongTensor 且在同一设备
        if not torch.is_tensor(log_seqs):
            log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=device)
        else:
            log_seqs = log_seqs.to(device).long()

        # item embedding
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # 用 torch.arange + 扩展来生成位置索引（避免 numpy）
        B, T = log_seqs.size()
        poss = torch.arange(1, T + 1, device=device).unsqueeze(0).expand(B, T)
        poss = poss * (log_seqs != 0).long()

        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # causal mask 在同一设备上创建
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=device)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        device = self.item_emb.weight.device
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(pos_seqs.to(device).long())
        neg_embs = self.item_emb(neg_seqs.to(device).long())

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        device = self.item_emb.weight.device
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(item_indices.to(device).long())
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
    # def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
    #     seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
    #     seqs *= self.item_emb.embedding_dim ** 0.5
    #     poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
    #     # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
    #     poss *= (log_seqs != 0)
    #     seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
    #     seqs = self.emb_dropout(seqs)

    #     tl = seqs.shape[1] # time dim len for enforce causality
    #     attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

    #     for i in range(len(self.attention_layers)):
    #         seqs = torch.transpose(seqs, 0, 1)
    #         if self.norm_first:
    #             x = self.attention_layernorms[i](seqs)
    #             mha_outputs, _ = self.attention_layers[i](x, x, x,
    #                                             attn_mask=attention_mask)
    #             seqs = seqs + mha_outputs
    #             seqs = torch.transpose(seqs, 0, 1)
    #             seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
    #         else:
    #             mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
    #                                             attn_mask=attention_mask)
    #             seqs = self.attention_layernorms[i](seqs + mha_outputs)
    #             seqs = torch.transpose(seqs, 0, 1)
    #             seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

    #     log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

    #     return log_feats

    # def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
    #     log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

    #     pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
    #     neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

    #     pos_logits = (log_feats * pos_embs).sum(dim=-1)
    #     neg_logits = (log_feats * neg_embs).sum(dim=-1)

    #     # pos_pred = self.pos_sigmoid(pos_logits)
    #     # neg_pred = self.neg_sigmoid(neg_logits)

    #     return pos_logits, neg_logits # pos_pred, neg_pred

    # def predict(self, user_ids, log_seqs, item_indices): # for inference
    #     log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

    #     final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

    #     item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

    #     logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    #     # preds = self.pos_sigmoid(logits) # rank same item list for different users

    #     return logits # preds # (U, I)