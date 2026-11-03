"""
model.py
-----------
Encoder-only citation model. The ENCODER reads the whole citation context and returns one vector per token.
Those token vectors are pooled into ONE context vector (mean over the real tokens, or the first token).
The pooled vector feeds two heads:
    query_proj : ONE citation query vector in the paper-embedding space (used to fetch the nearest papers)
    count_head : how many nearest papers to fetch (1..K)
There is no decoder and no language-model loss.
"""
# PyTorch base classes
import torch
# Neural network layers
import torch.nn as nn
# Functional ops such as normalize
import torch.nn.functional as F
# Loads the pretrained encoder (for SciEmbed this is a ModernBERT model)
from transformers import AutoModel


class CitationEncoder(nn.Module):
    # Configure the pretrained encoder and the citation-prediction heads.
    def __init__(self, model_name, embed_dim, max_slots, dropout, pooling):
        # Initialize nn.Module so PyTorch can track this model's layers
        super().__init__()
        # Maximum number of papers (K) to fetch per context
        self.K = max_slots
        # Size of one paper vector (d)
        self.d = embed_dim
        # How token vectors are pooled into one vector: "mean" or "cls"
        self.pooling = pooling
        # Fail early if the pooling name is not supported
        if pooling not in ("mean", "cls"):
            # Explain which values are allowed
            raise ValueError(f"pooling must be 'mean' or 'cls', got '{pooling}'")
        # Load the pretrained encoder weights and layers
        self.encoder = AutoModel.from_pretrained(model_name)
        # Hidden-state width of the encoder (768 for ModernBERT-base)
        H = self.encoder.config.hidden_size
        # Dropout applied to the pooled vector before the heads so that the model does not overfit to the training contexts
        self.dropout = nn.Dropout(dropout)
        # Project the vector from the encoder's hidden size to the paper-embedding size (d)
        self.query_proj = nn.Linear(H, embed_dim)
        # Count head: a single linear layer that predicts how many papers to fetch (1..K) for every context
        # It outputs one score for each possible citation count from 1 through K
        self.count_head = nn.Linear(H, max_slots)

    # input_ids and attention_mask have shape [G: batch_size, L: sequence_length]: tokenized contexts and their valid-token masks.
    # Returns one unit query vector per context and the count scores.
    def forward(self, input_ids, attention_mask):
        # Run the encoder: one updated vector per token, shape [G, L, H]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Token vectors of the last encoder layer
        h = outputs.last_hidden_state
        # Pool the token vectors into one vector per context
        if self.pooling == "cls":
            # Use the first token's vector as the summary, shape [G, H]
            pooled = h[:, 0]
        # Otherwise average over the real tokens
        else:
            # Mask with 1 for real tokens and 0 for padding, shape [G, L, 1]
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            # Sum the vectors of the real tokens, shape [G, H]
            summed = (h * mask).sum(dim=1)
            # Number of real tokens per context, shape [G, 1] (never zero)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            # Mean of the real token vectors, shape [G, H]
            pooled = summed / counts
        # Apply dropout to the pooled vector
        pooled = self.dropout(pooled)
        # Project to shape [G, d]
        projected = self.query_proj(pooled)
        # Normalize the query vector to unit length
        query = F.normalize(projected, dim=-1)
        # K scores representing the possible citation counts, shape [G, K]
        count_logits = self.count_head(pooled)
        # Return the query vector and the count scores
        return query, count_logits

    # query has shape [G, d]; paper_embs has shape [N, d] (unit vectors from the paper tower).
    # Returns, for each context, the indices of its k nearest papers, where k comes from count_head.
    @torch.no_grad()
    def retrieve(self, query, count_logits, paper_embs):
        # Best count score gives k in 1..K
        k_per_ctx = count_logits.argmax(dim=-1) + 1
        # Cosine similarity, since both sides are unit vectors
        sims = query @ paper_embs.T
        # Take the K nearest once, then cut each row to its own k
        top = sims.topk(self.K, dim=-1).indices
        # Start an empty list of results
        results = []
        # Visit every context with its own k
        for i, k in enumerate(k_per_ctx.tolist()):
            # Keep the k nearest papers of this context
            results.append(top[i, :k])
        # Return the nearest papers of every context
        return results