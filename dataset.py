"""
shared/data_prep/dataset.py
---------------------------
PyTorch Dataset + train/val/test split for the LCR citation model.

Each sample:
    context_text  : str                 — citing passage (input to CitationBART)
    cited_paper_id: int                 — global integer ID of the cited paper
                                          (index into feat_P / feat_PP / feat_PCP tensors)

Source file: shared/data_prep/all_contexts.json
    [
      {
        "context":     <str>,
        "cited_uri":   "https://citekg.org/resource/paper/<hash>",
        "citing_uri":  "https://citekg.org/resource/paper/<hash>",
        "citing_idx":  <int>
      },
      ...
    ]

Filters applied:
    - cited_uri must exist in node_index["paper"]  (already guaranteed by merge,
      but re-checked here for safety)
    - context must be non-empty after stripping

Split: deterministic random split seeded at 42
    train 80% / val 10% / test 10%
"""

import os
import json
import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer



# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class CitationRecord:
    context_text:      str
    cited_paper_id:    int   # global integer index into feat tensors
    cited_uri:         str   # kept for debugging / evaluation
    citing_uri:        str
    citation_type:     str   # "single" or "multiple"
    context_group_id:  int   # shared ID across records that come from the same citation marker
                             # (e.g. [1,2,3] → 3 records all sharing the same context_group_id)
                             # used by WholeGroupBatchSampler to keep them in the same batch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LCRDataset(Dataset): # list of pairs (context_text, cited_paper_id) 
    """
    Parameters
    ----------
    records : list[CitationRecord]  
    tokenizer : transformers tokenizer
    max_length : int
        Maximum token length for the tokenizer (BART: hard cap 1024).
    """

    def __init__( # store record "list of citation record" + tokenizer
        self,
        records: List[CitationRecord],
        tokenizer,
        max_length: int = 256, # BART's hard cap is 1024 but my contexts are short, so 256 is enough 
    ):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx): # Takes one CitationRecord, runs the tokenizer on context_text and returns a dict with input_ids, attention_mask, and cited_paper_id
        rec = self.records[idx] # called once per sample per epoch.

        enc = self.tokenizer(
            rec.context_text, # cotext text to be tokenized
            max_length=self.max_length, # max number of tokens to keep
            truncation=True, # truncate extra tokens if context_text is longer than max_length
            padding=False,          # collate_fn handles padding
            return_tensors=None,    # return plain lists; collate pads to batch max
        )

        return {
            "input_ids":         enc["input_ids"],        # batch of token sequences
            "attention_mask":    enc["attention_mask"],   # batch of masks (indicates which token to focus on and which to ignore(padded tokens))
            "cited_paper_id":    rec.cited_paper_id,      # batch of labels
            "context_group_id":  rec.context_group_id,   # used by collapse_groups (pool.py) to merge the records of one marker — not used by the loss
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def lcr_collate_fn(batch): 
    #prepare the data for the pytorch's DataLoader to handle variable-length sequences by padding them to the same length within a batch.
    # It receives a batch of variable-length sequences (context) from __getitem__, 
    # finds the longest sequence in the batch
    #  add zeros to sequences shorter than the longest sequence (to the attention mask as well as the input_ids)
    """
    Pads input_ids and attention_mask to the longest sequence in the batch.
    Returns:
        input_ids      : LongTensor [B, max_seq_len]
        attention_mask : LongTensor [B, max_seq_len]
        cited_paper_id : LongTensor [B]
    """
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids        = []
    attention_mask   = []
    cited_ids        = []
    context_group_ids = []

    for x in batch:
        seq_len = len(x["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(x["input_ids"] + [0] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        cited_ids.append(x["cited_paper_id"])
        context_group_ids.append(x["context_group_id"])

    return {
        "input_ids":        torch.tensor(input_ids,         dtype=torch.long),
        "attention_mask":   torch.tensor(attention_mask,    dtype=torch.long),
        "cited_paper_id":   torch.tensor(cited_ids,         dtype=torch.long),
        "context_group_id": torch.tensor(context_group_ids, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Builder 
# ---------------------------------------------------------------------------

def build_datasets(
    all_contexts_path: str,
    node_index_path:   str,
    tokenizer_name:    str  = "facebook/bart-base",
    max_length:        int  = 256,
    train_ratio:       float = 0.8,
    val_ratio:         float = 0.1,
    seed:              int   = 42,
    split_path:        Optional[str] = None,
) -> dict:
    """
    Loads all_contexts.json, filters, maps URIs → integer IDs,
    splits into train/val/test, and returns LCRDataset objects.

    Split behavior
    --------------
    If `split_path` is given and the file exists, the train/val/test
    citing_uri lists are loaded directly from it (see freeze_split.py) —
    the seed/shuffle logic below is skipped entirely, so the split cannot
    silently drift even if all_contexts.json / node_index.json change later.

    If `split_path` is given but the file does NOT exist yet, this function
    falls back to the seed-based shuffle-and-split below, and then WRITES
    the resulting citing_uri lists to `split_path` — so the very first run
    freezes the split for everything after it.

    If `split_path` is None, behavior is unchanged from before: the split is
    recomputed every call from (seed, current data files). This keeps old
    call sites working exactly as before if they don't pass split_path.

    Returns
    -------
    {
        "train": LCRDataset,
        "val":   LCRDataset,
        "test":  LCRDataset,
        "tokenizer": tokenizer,
        "n_papers":  int,        # total number of paper nodes (corpus + external)
    }
    """
    # --- Load node index ---> mapping for all papers from node_index.json by  {uri → int_id} 
    print(f"Loading node index from {node_index_path} ...")
    with open(node_index_path, encoding="utf-8") as f:
        node_index = json.load(f)
    paper_uri_to_id = node_index["paper"]   # {uri: int_id}
    n_papers = len(paper_uri_to_id)
    print(f"  {n_papers:,} paper nodes in KG")

    # --- Load and filter records ---> iterates every record and build a flat list of CitationRecord objects from all_contexts.json
    print(f"Loading contexts from {all_contexts_path} ...")
    with open(all_contexts_path, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  {len(raw):,} raw records")

    records = []
    skipped = 0
    for item in raw:
        cited_uri         = item.get("cited_uri", "")
        context_text      = item.get("context", "").strip()
        citing_uri        = item.get("citing_uri", "")
        citation_type     = item.get("citation_type", "")
        context_group_id  = item.get("context_group_id")
        # skip records with missing context, missing cited uri, missing citation_type,
        # or missing context_group_id (indicates file was not processed by add_context_group_id.py)
        if not context_text:
            skipped += 1
            continue
        if not citation_type:
            skipped += 1
            continue
        if context_group_id is None:
            skipped += 1
            continue
        if cited_uri not in paper_uri_to_id:
            skipped += 1
            continue

        records.append(CitationRecord(
            context_text     = context_text,
            cited_paper_id   = paper_uri_to_id[cited_uri],
            cited_uri        = cited_uri,
            citing_uri       = citing_uri,
            citation_type    = citation_type,
            context_group_id = context_group_id,
        ))

    print(f"  {len(records):,} records kept, {skipped:,} skipped")

    # --- Split: frozen file (preferred) or deterministic shuffle (fallback) ---
    #
    # Loading a frozen split_uris.json (produced once by freeze_split.py) makes
    # the train/val/test partition immune to later changes in all_contexts.json
    # / node_index.json — the exact same citing_uris always land in the same
    # bucket, so comparisons across experiments trained at different times stay
    # apples-to-apples. Without split_path (or on the very first run before the
    # file exists), we fall back to the original seed-based shuffle-and-split,
    # and — if split_path was given — persist that result so all future runs
    # pick up the frozen file from here on.
    used_frozen_split = False
    if split_path is not None and os.path.exists(split_path):
        print(f"Loading frozen split from {split_path} ...")
        with open(split_path, encoding="utf-8") as f:
            frozen = json.load(f)
        train_uris = set(frozen["train"])
        val_uris   = set(frozen["val"])
        test_uris  = set(frozen["test"])
        used_frozen_split = True
    else:
        citing_uris = list({r.citing_uri for r in records})
        rng = random.Random(seed)
        rng.shuffle(citing_uris)

        n_uris       = len(citing_uris)
        n_train_uris = int(n_uris * train_ratio)
        n_val_uris   = int(n_uris * val_ratio)

        train_uris = set(citing_uris[:n_train_uris])
        val_uris   = set(citing_uris[n_train_uris : n_train_uris + n_val_uris])
        test_uris  = set(citing_uris[n_train_uris + n_val_uris :])

        if split_path is not None:
            print(f"No frozen split found at {split_path} — computing it now "
                  f"and saving it so future runs are locked to it.")
            os.makedirs(os.path.dirname(os.path.abspath(split_path)) or ".", exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump({
                    "train": sorted(train_uris),
                    "val":   sorted(val_uris),
                    "test":  sorted(test_uris),
                    "meta": {
                        "seed": seed,
                        "train_ratio": train_ratio,
                        "val_ratio": val_ratio,
                        "n_kept": len(records),
                        "n_skipped": skipped,
                    },
                }, f, indent=2)

    train_records = [r for r in records if r.citing_uri in train_uris]
    val_records   = [r for r in records if r.citing_uri in val_uris]
    test_records  = [r for r in records if r.citing_uri in test_uris]

    split_source = "frozen file" if used_frozen_split else "seed-based shuffle"
    print(f"  Split ({split_source}) → train {len(train_records):,} / "
          f"val {len(val_records):,} / test {len(test_records):,}")

    # --- Tokenizer ---> load tokenizer and wrap each split train/val/test inside LCRDataset (a custom PyTorch Dataset class)
    print(f"Loading tokenizer ({tokenizer_name}) ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return {
        "train":     LCRDataset(train_records, tokenizer, max_length),
        "val":       LCRDataset(val_records,   tokenizer, max_length),
        "test":      LCRDataset(test_records,  tokenizer, max_length),
        "tokenizer": tokenizer,
        "n_papers":  n_papers,
    }