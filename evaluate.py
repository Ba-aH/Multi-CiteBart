"""
evaluate.py
--------------
Group-level evaluation: a paper's score is its cosine with the context's single query vector,
and a gold paper's rank = 1 + number of papers scoring higher. The count head decides how many
papers are fetched (the top-k), and is scored by count_acc.
"""
import math
from collections import defaultdict

import torch
from tqdm import tqdm

from pool import collapse_groups


# ---------------------------------------------------------------------------
# ranks_to_metrics: Evaluation metric calculation
#                  input: - dictionnary of input ranks => group_ranks = { group_id_1: [rank_of_gold1, rank_of_gold2, ...], 
#                                                                         group_id_2: [rank_of_gold1, ...],...}
#                  output: - dictionnary of averaged metrics {R@10,MRR,nDCG ...}
#
# ---------------------------------------------------------------------------
def ranks_to_metrics(group_ranks):
    k_values = [1, 5, 10, 20]                                 
    recall_hits = {k: 0.0 for k in k_values}                  # running sums, one per k
    mrr_sum, ndcg_sum, n_queries = 0.0, 0.0, 0
    for ranks in group_ranks.values():                        # one entry per context group
        n_papers = len(ranks)                                 # number of gold papers for a context group
        for k in k_values:
            denom = min(n_papers, k)                          # you cannot hit more than k papers in the top k
            hits = sum(1 for r in ranks if r <= k)            # gold papers inside the top k
            recall_hits[k] += hits / denom
        mrr_sum += sum(1.0 / r for r in ranks) / n_papers     # mean reciprocal rank inside the group
        ndcg_sum += sum(1.0 / math.log2(r + 1) for r in ranks) / n_papers
        n_queries += 1                                        # each GROUP counts as one query
    if n_queries == 0:
        return {}
    metrics = {f"Recall@{k}": recall_hits[k] / n_queries for k in k_values}
    metrics.update({"MRR": mrr_sum / n_queries, "nDCG@10": ndcg_sum / n_queries, "n_queries": n_queries})
    return metrics


@torch.no_grad()
def evaluate(model, loader, pool, id2row, device, K):
    model.eval()
    group_ranks = defaultdict(list)                           # group id -> ranks of its gold papers
    n_groups = n_exact = 0
    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        grp = collapse_groups(batch, id2row, max_gold=None)   # keep ALL gold papers of every group
        if grp is None:
            continue
        input_ids = grp["input_ids"].to(device)                # [G, L] token ids, moved to the device
        attention_mask = grp["attention_mask"].to(device)      # [G, L] real-token mask, moved to the device
        query, count_logits = model(input_ids, attention_mask) # one query vector and the count scores per group
        n_hat = count_logits.argmax(dim=-1) + 1               # [G] predicted number of papers
        n_true = grp["n_gold"].to(device).clamp(max=K)        # [G] true number (capped to K)

        fused = query.float() @ pool.T                        # [G, M] cosine of the single query with every paper

        gold_rows = grp["gold_rows"].to(device)               # [G, Kg] (-1 = padding)
        s_gold = fused.gather(1, gold_rows.clamp(min=0))      # score of every gold paper
        ranks = ((fused[:, None, :] > s_gold[:, :, None]).sum(dim=2) + 1).cpu().tolist()
        valid = (gold_rows >= 0).cpu().tolist()
        for g, gid in enumerate(grp["group_ids"]):
            group_ranks[gid] += [r for r, ok in zip(ranks[g], valid[g]) if ok]

        n_groups += len(grp["group_ids"])
        n_exact += int((n_hat == n_true).sum())

    metrics = ranks_to_metrics(group_ranks)
    metrics["count_acc"] = n_exact / max(n_groups, 1)         # share of contexts with exactly the right count
    return metrics