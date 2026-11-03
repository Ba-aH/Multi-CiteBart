"""
losses.py
------------
Final loss = 0.5 * count loss (how many papers?) + 0.5 * alignment loss (which papers?).
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# query_alignment_loss
# the model has ONE query vector per context.
# Multi-positive InfoNCE: the single query is pulled toward ALL of its gold papers at once.
# kg = number of gold papers for the context group 
# G = number of context groups in the batch
# M = number of candidate papers in the pool
# query [G, d] = single query vector for every context group (one querry per context / preventing the model from cheating by using multiple queries to cover all gold papers (identical [CLS] token for every context group))
# gold_rows [G, Kg] = pool rows of the gold papers for every context group
# train_embs [M, d] = frozen pool of candidate papers
# tau = temperature for InfoNCE 
# ---------------------------------------------------------------------------
def query_alignment_loss(query, gold_rows, train_embs, tau):
    dev = query.device                                        # where the vectors live
    gold_rows = gold_rows.to(dev)                             # fetch the gold rows to the same device as the query vectors
    valid = gold_rows >= 0                                    # valid store
    logits = query.float() @ train_embs.T / tau               # [G, M] similarity to every frozen paper
    log_p = F.log_softmax(logits, dim=1)                      # [G, M] log-probability of every paper
    log_p_gold = log_p.gather(1, gold_rows.clamp(min=0))      # [G, Kg] log-probability of every gold paper
    per_group = -(log_p_gold * valid).sum(1) / valid.sum(1)   # [G] mean over the gold papers of the group
    return per_group.mean(), int(valid.sum())                 # mean InfoNCE over all groups


# ---------------------------------------------------------------------------
# total_loss
#
# count     : cross-entropy loss for the count head (how many papers to fetch)
# align     : InfoNCE loss for the single query (which papers to fetch)
# total     : 0.5 * count + 0.5 * align
# ---------------------------------------------------------------------------
def total_loss(query, count_logits, grp, pool, tau):
    n_gold = grp["n_gold"].to(count_logits.device)            # true number of gold papers for every context group
    K = count_logits.size(1) # count logits = [G, K] where K = max number of papers to fetch and G = number of context groups in the batch
    count = F.cross_entropy(count_logits.float(), n_gold.clamp(max=K) - 1) # cross-entropy loss for the count head (how many papers to fetch)
    align, _ = query_alignment_loss(query, grp["gold_rows"], pool, tau)
    return 0.5 * count + 0.5 * align, align, count