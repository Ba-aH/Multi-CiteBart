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
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration


class CitationBART(nn.Module):
    def __init__(self, model_name, embed_dim, max_slots, dropout):
        super().__init__()
        self.K, self.d = max_slots, embed_dim
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        H = self.bart.config.d_model                          # hidden size
        self.dropout = nn.Dropout(dropout)
        self.vec_head = nn.Linear(H, max_slots * embed_dim)   # [CLS] state -> K vectors
        self.count_mlp = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(H, max_slots))   # [CLS] state -> scores for 1..K papers

    # input_ids / attention_mask: [G, L] = <s> context </s> from the dataloader
    # Returns: vecs [G, K, d] (unit length), count_logits [G, K], lm_loss (0 when with_lm is False)
    def forward(self, input_ids, attention_mask, with_lm):
        dec_ids = input_ids[:, :-1] if with_lm else input_ids[:, :1]       # decoder input: <s> w1 ... (or just <s>)
        dec_mask = attention_mask[:, :dec_ids.size(1)]
        h = self.bart.model(input_ids=input_ids, attention_mask=attention_mask,
                            decoder_input_ids=dec_ids, decoder_attention_mask=dec_mask).last_hidden_state
        cls_h = self.dropout(h[:, 0])                         # the [CLS] position: the only state we read out
        vecs = F.normalize(self.vec_head(cls_h).view(-1, self.K, self.d), dim=-1)
        count_logits = self.count_mlp(cls_h)

        lm_loss = h.new_zeros(())
        if with_lm:                                           # next-token prediction on the context
            logits = self.bart.lm_head(h) + self.bart.final_logits_bias
            tgt = input_ids[:, 1:].masked_fill(attention_mask[:, 1:] == 0, -100)   # ignore padding
            lm_loss = F.cross_entropy(logits.transpose(1, 2).float(), tgt, ignore_index=-100)
        return vecs, count_logits, lm_loss
