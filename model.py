"""
model.py
--------
BART encoder-decoder. The ENCODER reads the whole citation context. The DECODER starts with the
[CLS] token (BART's <s>) and, being causal, only ever works from that token and the encoder output.
The decoder state at [CLS] feeds two heads:
    vec_head  : K citation vectors in the paper-embedding space (one per possible cited paper)
    count_mlp : how many of them to use (1..K)
The autoregressive part (decoder predicting the context word by word) is kept as an optional LM loss.
"""
import torch  
import torch.nn as nn  .
import torch.nn.functional as F  
from transformers import BartForConditionalGeneration  # Loads the pretrained BART encoder-decoder.


class CitationBART(nn.Module):
    # Configure the pretrained BART model and the citation-prediction heads.
    def __init__(self, model_name, embed_dim, max_slots, dropout):
        super().__init__()  # Initialize nn.Module so PyTorch can track this model's layers.
        self.K, self.d = max_slots, embed_dim  # Store the number of citation slots and each vector's size.
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)  # Load pretrained BART weights and layers.
        H = self.bart.config.d_model  # Get BART's hidden-state width for configuring the output heads.
        self.dropout = nn.Dropout(dropout)  # Regularize the decoder state before making predictions.
        self.vec_head = nn.Linear(H, max_slots * embed_dim)  # Project the decoder state into K citation vectors.
        # Produce one score for each possible citation count from 1 through K.
        self.count_mlp = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(H, max_slots))

    # input_ids and attention_mask have shape [G: batch_size, L: sequence_length]: tokenized contexts and their valid-token masks.
    # Returns K unit citation vectors per context, count scores, and optional next-token language-model loss.
    def forward(self, input_ids, attention_mask, with_lm):
        # For language-model training, feed all but the final token; otherwise, decode from the start token alone.
        dec_ids = input_ids[:, :-1] if with_lm else input_ids[:, :1]
        dec_mask = attention_mask[:, :dec_ids.size(1)]  # Keep the decoder mask aligned with its input tokens.
        # Run BART's encoder and decoder; the result has shape [G, decoder_length, H].
        h = self.bart.model(input_ids=input_ids, attention_mask=attention_mask,
                            decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask).last_hidden_state
        cls_h = self.dropout(h[:, 0])  # Use the decoder's first position, corresponding to BART's start token.
        # Reshape the projected output to [G, K, d] and normalize each citation vector to unit length.
        vecs = F.normalize(self.vec_head(cls_h).view(-1, self.K, self.d), dim=-1)
        count_logits = self.count_mlp(cls_h)  # Return K scores representing the possible citation counts.

        lm_loss = h.new_zeros(())  # Default to a scalar zero on the same device and with the same dtype as h.
        if with_lm:  # Optionally train BART to predict each context's next token.
            logits = self.bart.lm_head(h) + self.bart.final_logits_bias  # Convert decoder states to vocabulary scores.
            # Shift labels one token ahead and mark padding positions with -100 so the loss ignores them.
            tgt = input_ids[:, 1:].masked_fill(attention_mask[:, 1:] == 0, -100)
            # Cross-entropy expects class scores on dimension 1; ignore_index skips the marked padding labels.
            lm_loss = F.cross_entropy(logits.transpose(1, 2).float(), tgt, ignore_index=-100)
        return vecs, count_logits, lm_loss  # Return citation vectors, count scores, and the optional LM loss.
