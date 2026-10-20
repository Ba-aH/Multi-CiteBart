"""
losses.py
---------
Final loss = 0.5 * count loss (how many papers?) + 0.5 * alignment loss (which papers?).
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# assign_slots: must be adjusted
#
# context having multiple citation context (eg. mutliple citation slots) this function assign an ordered slots for those citations
# (cost[i][j] = -cosine(slot i, gold j)). Hungarian algorithm via scipy.
# Returns two lists (slot_idx, gold_idx): slot slot_idx[k] is assigned to gold gold_idx[k].
# ---------------------------------------------------------------------------
def assign_slots(cost):
    n = cost.shape[0]                                         # number of slots (= number of gold papers)
    if n == 1:                                                # trivial case: the only slot gets the only gold
        return [0], [0]
    rows, cols = linear_sum_assignment(cost)                  # Hungarian algorithm
    return rows.tolist(), cols.tolist()


# ---------------------------------------------------------------------------
# slot_alignment_loss
# the model learns the best combination to fill the slots

# ---------------------------------------------------------------------------
def slot_alignment_loss(vecs, gold_rows, train_embs, tau):
    dev = vecs.device                                         # where the vectors live
    pair_g, pair_i, pair_t = [], [], []                       # group index, slot index, target pool row
    gold_list = gold_rows.tolist()                           
    with torch.no_grad():                                     # matching is a discrete decision: no gradient
        for g in range(vecs.size(0)):                         # for every group in the batch
            golds = [x for x in gold_list[g] if x >= 0]       # its real gold rows (drop the -1 padding)
            n = len(golds)                                    # number of slots used = number of gold papers
            q = vecs[g, :n].float()                           # [n, d] its slot vectors
            e = train_embs[torch.tensor(golds, device=dev)]   # [n, d] its gold embeddings
            cost = (-(q @ e.T)).cpu().numpy()                 # [n, n] negative cosine similarity
            slot_idx, gold_idx = assign_slots(cost)           # best one-to-one assignment
            for i, j in zip(slot_idx, gold_idx):              # record every matched pair
                pair_g.append(g)
                pair_i.append(i)
                pair_t.append(golds[j])
    pair_g = torch.tensor(pair_g, device=dev)                 # [P]
    pair_i = torch.tensor(pair_i, device=dev)                 # [P]
    pair_t = torch.tensor(pair_t, device=dev)                 # [P] target rows (class labels)
    q_sel = vecs[pair_g, pair_i].float()                      # [P, d] the matched slot vectors (WITH gradient)
    logits = q_sel @ train_embs.T / tau                       # [P, M] similarity to every frozen paper
    return F.cross_entropy(logits, pair_t), len(pair_t)       # mean InfoNCE over all matched pairs


# ---------------------------------------------------------------------------
# total_loss
#
# count     : cross-entropy of the count MLP against the true number of gold papers (class = n - 1)
# alignment : slot vectors vs the whole pool (InfoNCE), slots matched to gold papers with Hungarian
# ---------------------------------------------------------------------------
def total_loss(vecs, count_logits, grp, pool, tau):
    n_gold = grp["n_gold"].to(count_logits.device)            # [G] true number of papers per context
    count = F.cross_entropy(count_logits.float(), n_gold - 1)
    align, _ = slot_alignment_loss(vecs, grp["gold_rows"], pool, tau)
    return 0.5 * count + 0.5 * align, align, count
