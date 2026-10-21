Read them in the order the data flows through the project, so each script uses only things you have already seen. Here are the nine files, one sentence each:

1. **`config.json`**: every setting in one place (paths, BART model name, `max_slots`, learning rates, `temperature`, `lambda_lm`), so you know which knobs exist before you read the code.
2. **`dataset.py`**: turns `all_contexts.json` into tokenized (context, cited paper) records and builds the frozen train/val/test split.
3. **`whole_group_sampler.py`**: builds batches in which all records of one citation marker travel together.
4. **`paper_tower.py`**: your Paper Module, which turns the metapath features of each paper into an L2-normalised embedding.
5. **`pool.py`**: runs the PaperTower once and prepares the embeddings for retrieval (drop zeros, merge duplicates, normalise). It also merges the records of one marker into a single example with several gold papers.
6. **`model.py`**: `CitationBART` reads the context with the encoder and uses the decoder state at [CLS] to produce K paper vectors and the predicted count.
7. **`losses.py`**: matches slots to gold papers with the Hungarian algorithm and combines the count loss and alignment loss as `0.5 × count + 0.5 × alignment`.
8. **`evaluate.py`**: uses the predicted count to pick how many slots to use, ranks every gold paper against the pool, and computes Recall@k, MRR, nDCG and count accuracy.
9. **`train.py`**: wires everything together (data, pool, model, optimizer, early stopping on validation Recall@10, final test).

Before you start, spend two minutes skimming `main()` in `train.py` to see the overall flow. Then read the files in the order above and finish with `train.py` in full.

The three files that matter most are `model.py`, `losses.py` and the first half of `pool.py`. `dataset.py`, `whole_group_sampler.py` and `paper_tower.py` are your existing code with only the comment fixes and a one-line tokenizer change.