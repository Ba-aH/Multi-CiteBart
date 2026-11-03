# LCR (encoder-only): multi-paper local citation recommendation

Given a citation context, the model predicts **how many** papers are cited and **which** ones.
It is a pretrained encoder (default: SciEmbed, a ModernBERT-base scientific embedder) trained against frozen paper embeddings from the trained Paper Module (PaperTower).
Retrieval is nearest-neighbour search by cosine similarity over those embeddings.

## How it works

1. **Frozen paper pool.** The trained PaperTower is run once over all candidate papers. The embeddings are already L2-normalised, so dot product = cosine.
2. **One example per citation marker.** A marker that cites several papers is stored as several records. They are batched together and merged into one example with several gold papers.
3. **Encoder reads the context.** The encoder reads the whole context and returns one vector per token.
4. **Pooling.** The token vectors are pooled into ONE context vector (`pooling = "mean"` averages the real tokens, `"cls"` takes the first token). That vector is the only thing read out:
   - `query_proj` turns it into ONE unit-length query vector in the paper-embedding space;
   - `count_mlp` turns it into scores for 1..`max_slots` papers.
5. **Loss.** `0.5 × count loss + 0.5 × alignment loss`
   - count loss: cross-entropy of the MLP against the true number of cited papers;
   - alignment loss: multi-positive InfoNCE of the single query vector against the whole pool, pulling it toward all gold papers of the context at once.
6. **No decoder.** There is no decoder and no language-model loss.
7. **Inference.** The MLP picks the number of papers k. Papers are ranked by cosine similarity with the query vector and the top k are returned.

## Files

| File | Purpose |
|---|---|
| `config.json` | All settings |
| `dataset.py` | Records, tokenization, collate function, frozen train/val/test split |
| `whole_group_sampler.py` | Batches that keep all records of one marker together |
| `paper_tower.py` | The Paper Module (PaperTower) |
| `pool.py` | Builds the frozen pool and merges the records of one marker into one example |
| `model.py` | `CitationEncoder` |
| `losses.py` | Alignment loss (multi-positive InfoNCE), count loss, total loss |
| `evaluate.py` | Group-level ranking metrics and count accuracy |
| `train.py` | Training loop, early stopping, final test |

## Requirements

```
pip install torch transformers numpy tqdm
```

The encoder's tokenizer and weights are downloaded from the Hugging Face hub on the first run. ModernBERT-based models need a recent `transformers` (4.48 or later).

## Data needed

Everything under `data_root` (set in `config.json`):

- `all_contexts.json`, `node_index.json`
- `split_uris.json` (created on the first run if it does not exist)
- `feat_<key>.pt` for every metapath key of the PaperTower checkpoint, plus `corpus_ids.pt` and `external_ids.pt`

Plus the trained two-tower checkpoint at `paper_ckpt`. It must contain `config` and `paper_tower`.

## Run

```
python train.py --config config.json
```

The best checkpoint (by validation Recall@10) and the final test metrics are written to `<output_dir>/<experiment_name>/`.

## Main config keys

| Key | Meaning |
|---|---|
| `model_name` | Encoder checkpoint (default `J0nasW/sciembed-full`; any `AutoModel` encoder works, e.g. `allenai/scibert_scivocab_uncased`) |
| `pooling` | `mean` (average of the real tokens) or `cls` (first token). SciEmbed was trained with mean pooling |
| `max_slots` | Maximum number of papers per context (K); size of the count head |
| `batch_size` | Number of context **groups** per batch (not records) |
| `temperature` | InfoNCE temperature |
| `lr_backbone` / `lr_head` | Learning rates for the encoder and for the new heads |
| `paper_ckpt` | Trained PaperTower checkpoint |

## Notes

- Gold papers per context are not truncated; only the count target is capped at `max_slots` (K).
- About 5% of the pool shares an embedding with another paper. They stay separate rows; `evaluate.py` ranks by strictly higher scores, so a tie never counts against the gold paper.
- `temperature` and the learning rates are starting values, not tuned.
- The project compiles, but it has not been run end to end yet. Do a one-batch smoke test before a full run.