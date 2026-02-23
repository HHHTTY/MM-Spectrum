import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(BaseModel, self).__init__()

    def forward(self, src, tgt, src_len, bptt=False, with_align=False):

        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        raise NotImplementedError

    def count_parameters(self, log=print):
        raise NotImplementedError


class NMTModel(BaseModel):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len, bptt=False, with_align=False):
        dec_in = tgt[:, :-1, :]
        enc_out, enc_final_hs, src_len = self.encoder(src, src_len)
        if not bptt:
            self.decoder.init_state(src, enc_out, enc_final_hs)
        dec_out, attns = self.decoder(dec_in, enc_out,
                                      src_len=src_len,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout, attention_dropout):
        self.encoder.update_dropout(dropout, attention_dropout)
        self.decoder.update_dropout(dropout, attention_dropout)

    def count_parameters(self, log=print):

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log('encoder: {}'.format(enc))
            log('decoder: {}'.format(dec))
            log('* number of parameters: {}'.format(enc + dec))
        return enc, dec


class LanguageModel(BaseModel):

    def __init__(self, encoder=None, decoder=None):
        super(LanguageModel, self).__init__(encoder, decoder)
        if encoder is not None:
            raise ValueError("LanguageModel should not be used"
                             "with an encoder")
        self.decoder = decoder

    def forward(self, src, tgt, src_len, bptt=False, with_align=False):
        if not bptt:
            self.decoder.init_state()
        dec_out, attns = self.decoder(
            src, enc_out=None, src_len=src_len,
            with_align=with_align
        )
        return dec_out, attns

    def update_dropout(self, dropout, attention_dropout):
        self.decoder.update_dropout(dropout, attention_dropout)

    def count_parameters(self, log=print):

        enc, dec = 0, 0
        for name, param in self.named_parameters():
            if "decoder" in name:
                dec += param.nelement()

        if callable(log):
            # No encoder in LM, seq2seq count formatting kept
            log("encoder: {}".format(enc))
            log("decoder: {}".format(dec))
            log("* number of parameters: {}".format(enc + dec))
        return enc, dec
