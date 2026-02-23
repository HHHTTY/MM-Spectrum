import torch
import torch.nn as nn


def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs=None,
                         batch_dim=1, batch_offset=None):

    offset = len(tgt_vocab)
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if src_vocabs is None:
            src_vocab = batch['src_ex_vocab'][b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch['indices'].data[batch_id]
            src_vocab = src_vocabs[index]

        for i in range(1, len(src_vocab)):
            sw = src_vocab.ids_to_tokens[i]
            ti = tgt_vocab[sw]
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.Tensor(blank).type_as(batch['indices'].data)
            fill = torch.Tensor(fill).type_as(batch['indices'].data)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores


class CopyGenerator(nn.Module):

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):

        _, slen = attn.size()
        batch, _, cvocab = src_map.size()

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map
        )
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):

        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss
