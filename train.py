"""
train.py
--------
python train.py --config config.json
"""
import os
import json
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_datasets, lcr_collate_fn
from whole_group_sampler import WholeGroupBatchSampler
from pool import load_paper_pool, collapse_groups
from losses import total_loss
from model import CitationBART
from evaluate import evaluate


def train_one_epoch(model, loader, optimizer, pool, id2row, device, cfg):
    model.train()
    lam_lm, total, n = cfg["lambda_lm"], 0.0, 0
    for batch in tqdm(loader, leave=False):
        grp = collapse_groups(batch, id2row, max_gold=cfg["max_slots"])   # one example per context group
        if grp is None:
            continue
        vecs, count_logits, lm_loss = model(grp["input_ids"].to(device), grp["attention_mask"].to(device),
                                            with_lm=lam_lm > 0)
        loss, align, count = total_loss(vecs, count_logits, grp, pool, cfg["temperature"])
        loss = loss + lam_lm * lm_loss                        # lambda_lm = 0 -> exactly 0.5*count + 0.5*align
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total, n = total + loss.item(), n + 1
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.load(open(parser.parse_args().config))

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")
    data_root = os.path.expanduser(cfg["data_root"])
    ckpt_dir = os.path.join(os.path.expanduser(cfg["output_dir"]), cfg["experiment_name"])
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- data: one batch = batch_size whole context groups ---
    datasets = build_datasets(
        all_contexts_path=os.path.join(data_root, "all_contexts.json"),
        node_index_path=os.path.join(data_root, "node_index.json"),
        tokenizer_name=cfg["model_name"], max_length=cfg["max_length"], seed=cfg["seed"],
        split_path=os.path.join(data_root, "split_uris.json"))

    def make_loader(split, shuffle):
        sampler = WholeGroupBatchSampler([r.context_group_id for r in datasets[split].records],
                                         groups_per_batch=cfg["batch_size"], seed=cfg["seed"], shuffle=shuffle)
        return DataLoader(datasets[split], batch_sampler=sampler, collate_fn=lcr_collate_fn, num_workers=4), sampler

    train_loader, train_sampler = make_loader("train", True)
    val_loader, _ = make_loader("val", False)
    test_loader, _ = make_loader("test", False)

    # --- frozen paper pool (precomputed embeddings) ---
    pool, id2row = load_paper_pool(os.path.expanduser(cfg["paper_embs"]), datasets["n_papers"], device)

    # --- model: BART body gets a small lr, the new heads a larger one ---
    model = CitationBART(cfg["model_name"], pool.size(1), cfg["max_slots"], cfg["dropout"]).to(device)
    heads = list(model.vec_head.parameters()) + list(model.count_mlp.parameters())
    optimizer = torch.optim.AdamW([{"params": model.bart.parameters(), "lr": cfg["lr_backbone"]},
                                   {"params": heads, "lr": cfg["lr_head"]}])

    # --- training with early stopping on validation Recall@10 ---
    best, bad = -1.0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        train_sampler.set_epoch(epoch)
        loss = train_one_epoch(model, train_loader, optimizer, pool, id2row, device, cfg)
        m = evaluate(model, val_loader, pool, id2row, device, cfg["max_slots"])
        print(f"Epoch {epoch:3d} | loss={loss:.4f} | VAL R@10={m['Recall@10']:.4f} MRR={m['MRR']:.4f} count_acc={m['count_acc']:.3f}")
        if m["Recall@10"] > best:
            best, bad = m["Recall@10"], 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break

    # --- final test with the best checkpoint ---
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device))
    test = evaluate(model, test_loader, pool, id2row, device, cfg["max_slots"])
    print("TEST:", {k: round(v, 4) for k, v in test.items()})
    json.dump(test, open(os.path.join(ckpt_dir, "test_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
