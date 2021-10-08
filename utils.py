import math

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


def scaled_dot_product(q, k, v, mask=None):
    # get hidden dimensionality for queries/keys
    d_k = q.size()[-1]

    # compute the query/key similarity score
    attn_logits = torch.matmul(q, k.transpose(-2,-1))
    # divide by the sqrt of the hidden dimensionality to maintain the appropriate variance of the attention values
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        # apply mask if available, we pad the sentences to the same length and mask out the padding tokens during the calculation of the attention values
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15) # set to a very low value
    
    # compute softmax and multiply with the value
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

    
if __name__ == "__main__":
    seq_len, d_k = 3, 2
    pl.seed_everything(42)
    q = torch.randn(seq_len, d_k)
    print("Q\n", q)
    k = torch.randn(seq_len, d_k)
    print("K\n", k)
    v = torch.randn(seq_len, d_k)
    print("V\n", v)
    values, attention = scaled_dot_product(q, k, v)
    print("Values\n", values)
    print("Attention\n", attention)
