"""
pool.py
-------
Everything that maps papers to rows of the frozen paper-embedding pool:
    build_candidate_index / load_frozen_paper_index - run the trained PaperTower ONCE over all papers
    load_paper_pool   - the frozen pool [M, d] + the map global paper id -> pool row
    collapse_groups   - turn a batch of (context, cited paper) records into context groups with gold pool rows
"""
import os

import torch
from tqdm import tqdm

from paper_tower import PaperTower   # your Paper Module script (was model.py), same folder


# ---------------------------------------------------------------------------
# Build frozen candidate index
# Runs PaperTower over all candidate papers ONCE, before training starts.
# The resulting matrix is fixed for the entire run —
# the PaperTower is frozen (never trained here), so the index never changes.
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_candidate_index(paper_tower, all_paper_feats, candidate_ids, device):
    # Save training state so we can restore it after — eval() disables dropout
    # which is required for deterministic embeddings during evaluation
    was_training = paper_tower.training
    paper_tower.eval()

    embs = []
    ids  = candidate_ids.tolist()

    # Encode all candidate papers in batches of 512 
    # The 512 embeddings are then moved to CPU memory to free up GPU memory for the next chunk to avoid GPU memory overflow
    # append the CPU embeddings to a python list and concatenate all chunks to form the final candidate embedding matrix
    for start in tqdm(range(0, len(ids), 512), desc="  Building index", leave=False):
        batch_ids   = ids[start : start + 512]
        # Look up precomputed metapath features for this chunk from the fixed disk tensors
        batch_feats = {k: v[batch_ids].to(device) for k, v in all_paper_feats.items()}
        # Move embeddings back to CPU immediately to free GPU memory
        embs.append(paper_tower(batch_feats).cpu())

    # Restore training state before returning
    paper_tower.train(was_training)

    # Concatenate all batches of embeddings into a single tensor and return it
    return torch.cat(embs, dim=0)


# ---------------------------------------------------------------------------
# load_frozen_paper_index
#
# Rebuilds your trained PaperTower from its checkpoint and runs it ONCE over all candidate papers.
# These embeddings are the fixed "answer sheet" the query vector must match.
# Returns: cand_embs [N, d] (CPU), cand_ids [N] (global ids), n_total (size of the global-id space).
# ---------------------------------------------------------------------------
def load_frozen_paper_index(cfg: dict, device):
    ckpt = torch.load(os.path.expanduser(cfg["paper_ckpt"]), map_location="cpu")   # saved two-tower checkpoint
    pcfg = ckpt["config"]                                     # hyperparameters it was trained with
    data_root = os.path.expanduser(cfg["data_root"])          # folder with feature / id files

    feats = {k: torch.load(os.path.join(data_root, f"feat_{k}.pt"), map_location="cpu")
             for k in pcfg["feat_keys"]}                      # metapath feature matrices, row i = paper id i
    cand_ids = torch.cat([                                    # candidate pool = corpus + external papers
        torch.load(os.path.join(data_root, "corpus_ids.pt"),   map_location="cpu"),
        torch.load(os.path.join(data_root, "external_ids.pt"), map_location="cpu"),
    ]).unique()                                               # remove duplicates (and sort)

    tower = PaperTower(                                       # rebuild with EXACTLY the training hyperparameters
        feat_keys=pcfg["feat_keys"], nfeat=768, num_heads=pcfg["num_heads"],
        hidden=pcfg["hidden"], embed_dim=pcfg["embed_dim"], n_fp_layers=pcfg["n_fp_layers"],
        dropout=pcfg["dropout"], input_drop=pcfg["input_drop"], att_drop=pcfg["att_drop"],
        act=pcfg["act"], residual=pcfg["residual"], use_mlp=pcfg["use_mlp"],
    ).to(device)
    tower.load_state_dict(ckpt["paper_tower"])                # trained weights

    cand_embs = build_candidate_index(tower, feats, cand_ids, device)   # [N, d] on CPU
    n_total = next(iter(feats.values())).size(0)              # number of paper ids

    del tower                                                 # free the GPU memory
    torch.cuda.empty_cache()
    return cand_embs, cand_ids, n_total


# ---------------------------------------------------------------------------
# load_paper_pool
#
# Takes the paper embeddings produced by the trained PaperTower and builds a frozen pool of candidate papers.
# The pool is a fixed [M, d] matrix of L2-normalized vectors,
# where M = number of candidate papers and d = embedding dimension.
# ---------------------------------------------------------------------------
def load_paper_pool(cfg, device):
    embs, ids, n_total = load_frozen_paper_index(cfg, device)   # [N, d] CPU, [N], int
    id2row = torch.full((n_total,), -1, dtype=torch.long)     # global paper id -> pool row (-1 = absent)
    id2row[ids] = torch.arange(len(ids))
    return embs.to(device), id2row


# ---------------------------------------------------------------------------
# collapse_groups
#
# Turns a batch of individual citation records into a batch of context groups. 
#  
# Records of the same group share the same context text, so we keep the first record's tokens and
# gather the cited papers of all its records into a padded list of pool rows.
#
# Arguments:
#   batch    : dict from the dataloader, with keys "context_group_id", "cited_paper_id", "input_ids", "attention_mask"
#   id2row   : CPU LongTensor mapping global paper id -> pool row (-1 = absent) to find the pool rows of the cited papers inside the batch
#   max_gold : keep at most this many gold papers per group 
# Returns None if no group has any gold papers, otherwise a dict with keys:
#   group_ids      : python list of group ids (one per context)
#   input_ids      : [G, L] LongTensor of tokenized contexts (one per group)
#   attention_mask : [G, L] LongTensor of valid-token masks
#   gold_rows      : [G, Kg] LongTensor of pool rows of the gold papers (padded with -1)
#   n_gold         : [G] LongTensor of how many gold papers each group has (1..Kg)
# ---------------------------------------------------------------------------
def collapse_groups(batch, id2row, max_gold=None):
    group_ids = batch["context_group_id"].tolist()            # group id of every record
    cited_rows = id2row[batch["cited_paper_id"]].tolist()     # global paper id -> pool row, per record
    first, golds = {}, {}                                     # first[g] = first record of group g, golds[g] = list of pool rows of its cited papers
                                                              
    for r, (g, row) in enumerate(zip(group_ids, cited_rows)): # walk over all records in the batch g is the group id, row is the pool row of the cited paper (-1 = doesn't exist in pool)
        if g not in first:                                    # if this is the first record of group g, remember it and store its gold papers
            first[g] = r                                      # record index of the first record of group g
            golds[g] = []                                     # initialize the list of gold papers for this group
        
        if row >= 0 and row not in golds[g]:                  # if the cited paper exists in the pool and is not already in the list of gold papers for this group then add it to the list of gold papers for this group
            golds[g].append(row)                              # adding the pool row of the cited paper to the list of gold papers for this group
    keep = [g for g in first if golds[g]]                     # keep only the groups that have at least one gold paper (one cited paper that exists in the pool)                     
                                                              # groups that have at least one gold paper
    if not keep:                                              # if there is no group with any gold papers, return None
        return None
    
    if max_gold is not None:                                  # capping the number of gold papers per group to max_gold if specified else keep all gold papers for each group and return them in the golds dictionary
        golds = {g: v[:max_gold] for g, v in golds.items()}   # dictionary storing the gold papers for each group, capped at max_gold
    first_idx = torch.tensor([first[g] for g in keep], dtype=torch.long)   # which record to take tokens from
    kg = max(len(golds[g]) for g in keep)                     # widest gold list in this batch
    gold_rows = torch.full((len(keep), kg), -1, dtype=torch.long)          # [G, Kg] padded with -1
    for i, g in enumerate(keep):                              # fill each group's gold rows
        gold_rows[i, : len(golds[g])] = torch.tensor(golds[g])
    return {
        "group_ids": keep,                                    # python list of group ids
        "input_ids": batch["input_ids"][first_idx],           # [G, L] one context per group
        "attention_mask": batch["attention_mask"][first_idx], # [G, L]
        "gold_rows": gold_rows,                               # [G, Kg]
        "n_gold": (gold_rows >= 0).sum(dim=1),                # [G] number of gold papers per group
    }