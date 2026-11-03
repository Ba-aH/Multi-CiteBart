"""
train.py
--------
python train.py --config config.json
"""
# Standard library: file paths
import os
# Standard library: read/write JSON
import json
# Standard library: Python RNG
import random
# Standard library: command-line arguments
import argparse

# NumPy for its RNG seeding
import numpy as np
# PyTorch for the model and training
import torch
# DataLoader wraps a dataset into batches
from torch.utils.data import DataLoader
# Progress bar for the training loop
from tqdm import tqdm

# Dataset builder and collate function
from dataset import build_datasets, lcr_collate_fn
# Sampler that keeps whole context groups in the same batch
from whole_group_sampler import WholeGroupBatchSampler
# Paper-pool loader and group-merging helper
from pool import load_paper_pool, collapse_groups
# Combined training loss
from losses import total_loss
# Encoder-only citation model
from model import CitationEncoder
# Validation/test evaluation function
from evaluate import evaluate


def train_one_epoch(model, loader, optimizer, pool, id2row, device, cfg):
    """Train for one pass over the grouped training data and return mean batch loss."""
    # Training mode: dropout is active (weights only change in optimizer.step() below)
    model.train()
    temperature = cfg["temperature"]

    # Running sum of batch losses (used for the epoch mean)
    total = 0.0
    # Number of batches actually processed
    n = 0
    # Iterate over batches with a progress bar
    for batch in tqdm(loader, leave=False):
        # A citation marker can have one dataset record per cited paper. Merge those
        # sibling records so the model sees one context with all of its gold papers.
        # Merge sibling records into one context with its gold pool rows (gold_rows) and their count (n_gold)
        grp = collapse_groups(batch, id2row, max_gold=None)
        # Skip batches that produced no usable group
        if grp is None:
            # Move on to the next batch
            continue

        
        # Token ids of the contexts, moved to the device
        input_ids = grp["input_ids"].to(device)

        # Mask marking real tokens (1) vs padding (0), moved to the device
        attention_mask = grp["attention_mask"].to(device)

        # FORWARD PASS: the encoder reads all context tokens; the token vectors are pooled into one
        # vector that feeds query_proj (query) and count_head (count_logits)
        query, count_logits = model(input_ids, attention_mask)

        # Combine retrieval alignment and citation-count prediction.
        # LOSS = 0.5 * count (cross-entropy vs true number of papers) + 0.5 * alignment
        # (multi-positive InfoNCE of the query against the frozen pool); align and count are returned for logging
        loss, align, count = total_loss(query, count_logits, grp, pool, temperature)

        # Update model parameters from this batch. Clipping limits unusually large
        # gradients, which can destabilize fine-tuning of the pretrained backbone.
        # Clear gradients left from the previous batch (otherwise they accumulate)
        optimizer.zero_grad()
        # BACKWARD PASS: compute the gradient of the loss for every trainable weight
        loss.backward()
        # All model weights whose gradients will be clipped
        parameters = model.parameters()
        # Largest allowed total gradient norm
        max_norm = 1.0
        # Rescale gradients if their total norm exceeds max_norm
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        # WEIGHT UPDATE: AdamW moves each weight a small step that lowers the loss (this is where the model learns)
        optimizer.step()
        # This batch's loss as a plain Python number
        batch_loss = loss.item()
        # Add this batch's loss to the running total
        total = total + batch_loss
        # Count this batch
        n = n + 1
    # Guard against division by zero if every batch was skipped.
    # Number of batches, at least 1
    n_batches = max(n, 1)
    # Mean batch loss over the epoch
    mean_loss = total / n_batches
    # Return the mean batch loss over the epoch
    return mean_loss


def main():
    # Load all run settings from one JSON file so experiments are reproducible.
    # Create the command-line parser
    parser = argparse.ArgumentParser()
    # Require a --config path argument
    parser.add_argument("--config", required=True)
    # Parse the command-line arguments
    args = parser.parse_args()
    # Path of the config file
    config_path = args.config
    # Open the config file
    with open(config_path) as config_file:
        # Load the settings dict from the JSON file
        cfg = json.load(config_file)

    # Seed Python, NumPy, and PyTorch to make data order and model initialization
    # repeatable; use the requested GPU when available and otherwise fall back to CPU.
    # Read the seed once
    seed = cfg["seed"]
    # Seed Python's RNG
    random.seed(seed)
    # Seed NumPy's RNG
    np.random.seed(seed)
    # Seed PyTorch's CPU RNG
    torch.manual_seed(seed)
    # Seed PyTorch's RNG on all GPUs
    torch.cuda.manual_seed_all(seed)
    # Check whether a GPU is available
    gpu_available = torch.cuda.is_available()
    # Use the configured GPU if available
    if gpu_available:
        # Device name such as "cuda:0"
        device_name = f"cuda:{cfg['gpu']}"
    # Otherwise fall back to the CPU
    else:
        # CPU device name
        device_name = "cpu"
    # Create the torch device object
    device = torch.device(device_name)
    # Data folder path as written in the config
    data_root_raw = cfg["data_root"]
    # Expand "~" in the data folder path
    data_root = os.path.expanduser(data_root_raw)
    # Output folder path as written in the config
    output_dir_raw = cfg["output_dir"]
    # Expand "~" in the output folder path
    output_dir = os.path.expanduser(output_dir_raw)
    # Checkpoint folder: output_dir/experiment_name
    ckpt_dir = os.path.join(output_dir, cfg["experiment_name"])
    # Create the checkpoint folder if missing
    os.makedirs(ckpt_dir, exist_ok=True)
    # Path where the best model weights are saved and reloaded
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    # Build the train/validation/test datasets. A batch is made of whole context
    # groups, so records for a multi-paper citation stay together for later merging.
    # Path of the contexts file
    all_contexts_path = os.path.join(data_root, "all_contexts.json")
    # Path of the paper node index file
    node_index_path = os.path.join(data_root, "node_index.json")
    # Path of the frozen train/val/test split file
    split_path = os.path.join(data_root, "split_uris.json")
    # Encoder model name (also used for its tokenizer)
    model_name = cfg["model_name"]
    # Maximum number of tokens kept per context
    max_length = cfg["max_length"]
    # Build the train/val/test datasets (one record per (context, cited paper), tokenized by the encoder's tokenizer)
    datasets = build_datasets(
        all_contexts_path=all_contexts_path,
        node_index_path=node_index_path,
        tokenizer_name=model_name,
        max_length=max_length,
        seed=seed,
        split_path=split_path,
    )

    # Number of context groups per batch
    batch_size = cfg["batch_size"]

    def make_loader(split, shuffle):
        # The sampler batches groups rather than individual records. Shuffling is
        # enabled only for training; validation and test order stays deterministic.
        # Dataset of this split
        split_dataset = datasets[split]
        # Citation records of this split
        records = split_dataset.records
        # Start an empty list of group IDs
        group_ids = []
        # Visit every record of this split
        for r in records:
            # Group ID of every record in this split
            group_ids.append(r.context_group_id)
        # Batch sampler that keeps whole groups together
        sampler = WholeGroupBatchSampler(
            group_ids,
            groups_per_batch=batch_size,
            seed=seed,
            shuffle=shuffle,
        )
        # DataLoader driven by that sampler (collate pads each batch to its longest sequence)
        loader = DataLoader(
            split_dataset,
            batch_sampler=sampler,
            collate_fn=lcr_collate_fn,
            num_workers=4,
        )
        # Return both the loader and its sampler
        return loader, sampler

    # Training loader (shuffled) and its sampler
    train_loader, train_sampler = make_loader("train", True)
    # Validation loader (deterministic); sampler not needed
    val_loader, _ = make_loader("val", False)
    # Test loader (deterministic); sampler not needed
    test_loader, _ = make_loader("test", False)

    # Build the frozen paper vectors used as retrieval candidates by running the trained
    # PaperTower (paper_ckpt) once over all candidate papers. The pool is
    # fixed during training; id2row maps paper IDs in the data to rows in this pool.
    # Build the paper embedding pool [M, d] (frozen, never trained here) and the ID-to-row map
    pool, id2row = load_paper_pool(cfg, device)

    # Fine-tune the pretrained encoder backbone conservatively while allowing the new
    # retrieval and count-prediction heads to learn faster from scratch.
    # Size of one paper vector (d)
    embed_dim = pool.size(1)
    # Maximum number of papers per context (K)
    max_slots = cfg["max_slots"]
    # Dropout probability of the heads
    dropout = cfg["dropout"]
    # How token vectors are pooled into one context vector ("mean" or "cls")
    pooling = cfg["pooling"]
    # Build CitationEncoder (pretrained encoder + query_proj and count_head heads)
    model = CitationEncoder(model_name, embed_dim, max_slots, dropout, pooling)
    # Move the model to the device
    model = model.to(device)
    # Parameters of the query head
    query_proj_params = list(model.query_proj.parameters())
    # Parameters of the count head
    count_head_params = list(model.count_head.parameters())
    # Collect the parameters of the two new heads (query_proj and count_head)
    heads = query_proj_params + count_head_params
    # Parameter group for the encoder with its small learning rate
    backbone_group = {"params": model.encoder.parameters(), "lr": cfg["lr_backbone"]}
    # Parameter group for the heads with their larger learning rate
    head_group = {"params": heads, "lr": cfg["lr_head"]}
    # AdamW with a small LR for the encoder (lr_backbone) and a larger LR for the heads (lr_head)
    optimizer = torch.optim.AdamW([backbone_group, head_group])

    # Train epoch by epoch and use validation Recall@10 to select the checkpoint.
    # Stop after `patience` consecutive epochs without an improvement.
    # Number of epochs allowed
    num_epochs = cfg["epochs"]
    # End of the epoch range (range end is exclusive, so epochs run 1..num_epochs)
    stop_epoch = num_epochs + 1
    # Epochs without improvement allowed before stopping
    patience = cfg["patience"]
    # Best validation Recall@10 seen so far
    best = -1.0
    # Consecutive epochs without improvement
    bad = 0
    # Loop over epochs, numbered from 1 (one epoch = one full pass over the training batches)
    for epoch in range(1, stop_epoch):
        # A new deterministic group order is generated for each training epoch.
        # Reseed the sampler's shuffle for this epoch
        train_sampler.set_epoch(epoch)
        # Train one epoch (weights change here) and get the mean loss
        loss = train_one_epoch(model, train_loader, optimizer, pool, id2row, device, cfg)
        # Evaluate on validation (no weight updates): ranking metrics and count accuracy
        m = evaluate(model, val_loader, pool, id2row, device, max_slots)
        # Validation Recall@10 of this epoch
        recall10 = m["Recall@10"]
        # Validation mean reciprocal rank of this epoch
        mrr = m["MRR"]
        # Validation accuracy of the predicted number of papers
        count_acc = m["count_acc"]
        # Log the epoch's loss and validation metrics
        print(f"Epoch {epoch:3d} | loss={loss:.4f} | VAL R@10={recall10:.4f} MRR={mrr:.4f} count_acc={count_acc:.3f}")
        # Check whether validation Recall@10 improved
        if recall10 > best:
            # Record the new best score
            best = recall10
            # Reset the no-improvement counter
            bad = 0
            # Keep the best validation checkpoint rather than the final epoch's weights.
            # Current model weights
            state = model.state_dict()
            # Save the model weights as the best checkpoint
            torch.save(state, best_path)
        # No improvement this epoch
        else:
            # Increment the no-improvement counter
            bad += 1
            # Stop if patience is exhausted
            if bad >= patience:
                # Leave the training loop (early stopping)
                break

    # Report test metrics only after restoring the checkpoint selected on validation.
    # Load the best checkpoint's weights from disk
    best_state = torch.load(best_path, map_location=device)
    # Put those weights into the model
    model.load_state_dict(best_state)
    # Evaluate on the test set (same procedure as validation, run once)
    test = evaluate(model, test_loader, pool, id2row, device, max_slots)
    # Start an empty dict for the rounded metrics
    rounded_test = {}
    # Visit every test metric
    for k, v in test.items():
        # Round the metric to 4 decimals
        rounded_test[k] = round(v, 4)
    # Print the test metrics rounded to 4 decimals
    print("TEST:", rounded_test)
    # Persist the metrics alongside the checkpoint for later comparison.
    # Path of the metrics file
    metrics_path = os.path.join(ckpt_dir, "test_metrics.json")
    # Open the metrics file for writing
    with open(metrics_path, "w") as metrics_file:
        # Write the metrics as indented JSON
        json.dump(test, metrics_file, indent=2)


# Run main() only when executed as a script
if __name__ == "__main__":
    # Start the program
    main()