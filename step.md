Read them in the order the data flows through the project, so each script uses only things you have already seen. Here are the nine files, one sentence each:

1. **`config.json`**: every setting in one place (paths, encoder model name, `pooling`, `max_slots`, learning rates, `temperature`), so you know which knobs exist before you read the code.
2. **`dataset.py`**: turns `all_contexts.json` into tokenized (context, cited paper) records and builds the frozen train/val/test split.
3. **`whole_group_sampler.py`**: builds batches in which all records of one citation marker travel together.
4. **`paper_tower.py`**: your Paper Module, which turns the metapath features of each paper into an L2-normalised embedding.
5. **`pool.py`**: runs the PaperTower once from `paper_ckpt` to build the frozen paper-embedding pool and the paper-id → pool-row map. It also merges the records of one marker into a single example with several gold papers.
6. **`model.py`**: `CitationEncoder` reads the context with a pretrained encoder, pools the token vectors into one vector, and produces one query vector and the predicted count.
7. **`losses.py`**: aligns the single query with all its gold papers (multi-positive InfoNCE) and combines it with the count loss as `0.5 × count + 0.5 × alignment`.
8. **`evaluate.py`**: ranks every gold paper against the pool by cosine with the query, and computes Recall@k, MRR, nDCG and count accuracy (predicted count vs true count).
9. **`train.py`**: wires everything together (data, pool, model, optimizer, early stopping on validation Recall@10, final test).

Before you start, spend two minutes skimming `main()` in `train.py` to see the overall flow. Then read the files in the order above and finish with `train.py` in full.

The three files that matter most are `model.py`, `losses.py` and the first half of `pool.py`. `dataset.py`, `whole_group_sampler.py` and `paper_tower.py` are your existing code with only the comment fixes and a one-line tokenizer change.