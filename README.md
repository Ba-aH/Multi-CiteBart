# LCR-BART: multi-paper local citation recommendation

Given a citation context, the model predicts **how many** papers are cited and **which** ones.
It is a BART encoder-decoder trained against frozen paper embeddings from the trained Paper Module (PaperTower).
Retrieval is nearest-neighbour search by cosine similarity over those embeddings.

## How it works

1. **Frozen paper pool.** The trained PaperTower is run once over all candidate papers. The embeddings are already L2-normalised, so dot product = cosine.
2. **One example per citation marker.** A marker that cites several papers is stored as several records. They are batched together and merged into one example with several gold papers.
3. **Encoder reads the context.** The BART encoder reads the whole context.
4. **Decoder reads out the [CLS] state.** The decoder starts from `<s>` (the [CLS] token) and attends to the encoder output. Its hidden state at that single position is the only thing read out:
   - `vec_head` turns it into `max_slots` paper vectors (one per possible cited paper);
   - `count_mlp` turns it into scores for 1..`max_slots` papers.
5. **Loss.** `0.5 × count loss + 0.5 × alignment loss`
   - count loss: cross-entropy of the MLP against the true number of cited papers;
   - alignment loss: InfoNCE of the slot vectors against the whole pool, with slots matched to gold papers one-to-one by the Hungarian algorithm.
6. **Optional LM loss.** The decoder can also be trained to predict the context word by word (`lambda_lm > 0`). It is off by default (`lambda_lm = 0`), because the encoder already sees the same text and the task is nearly trivial.
7. **Inference.** The MLP picks the number of slots n. A paper's score is its maximum cosine similarity over the first n slot vectors, and papers are ranked by that score.

## Files

| File | Purpose |
|---|---|
| `config.json` | All settings |
| `dataset.py` | Records, tokenization, collate function, frozen train/val/test split |
| `whole_group_sampler.py` | Batches that keep all records of one marker together |
| `paper_tower.py` | The Paper Module (PaperTower) |
| `pool.py` | Builds the frozen pool and merges the records of one marker into one example |
| `model.py` | `CitationBART` |
| `losses.py` | Hungarian matching, alignment loss, total loss |
| `evaluate.py` | Group-level ranking metrics and count accuracy |
| `train.py` | Training loop, early stopping, final test |

## Requirements

```
pip install torch transformers scipy numpy tqdm
```

The `facebook/bart-base` tokenizer and weights are downloaded from the Hugging Face hub on the first run.

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
| `model_name` | Encoder-decoder checkpoint (`facebook/bart-base`; `facebook/bart-large` also works) |
| `max_slots` | Maximum number of papers per context (K) |
| `batch_size` | Number of context **groups** per batch (not records) |
| `temperature` | InfoNCE temperature |
| `lambda_lm` | Weight of the optional LM loss (0 = off) |
| `lr_backbone` / `lr_head` | Learning rates for BART and for the new heads |
| `paper_ckpt` | Trained PaperTower checkpoint |

## Notes

- `max_slots` caps the gold papers per context during training; contexts with more are truncated.
- About 5% of the pool shares an embedding with another paper. They stay separate rows; `evaluate.py` ranks by strictly higher scores, so a tie never counts against the gold paper.
- `temperature` and the learning rates are starting values, not tuned.
- The project compiles, but it has not been run end to end yet. Do a one-batch smoke test before a full run.