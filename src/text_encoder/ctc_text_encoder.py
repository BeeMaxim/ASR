import re
from string import ascii_lowercase

from collections import defaultdict

import torch
from torchaudio.models.decoder import ctc_decoder

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()
    
    def expand_and_merge(self, dp, probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(probs):
            cur = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur:
                    new_prefix = prefix
                else:
                    if cur != self.EMPTY_TOK:
                        new_prefix = prefix + cur
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur)] += v * next_token_prob
        return new_dp
    
    def truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

    def ctc_decode(self, log_probs, beam_search=False) -> str:
        if beam_search:
            dp = {
                ('', self.EMPTY_TOK): 1.0
            }
            for prob in log_probs.cpu().exp().numpy():
                dp = self.expand_and_merge(dp, prob)
                dp = self.truncate_paths(dp, 25)
                # print(dp)
            # print(self.truncate_paths(dp, 1))
            return list(self.truncate_paths(dp, 1).keys())[0][0]
            
            #decoder = ctc_decoder(lexicon=None, tokens=self.vocab, beam_size=50, blank_token='', sil_token='z')
            # result = decoder(log_probs.unsqueeze(0).cpu())[0][0]
            
            # print(result.tokens)

            # return self.decode(result.tokens)
        last = -1
        tokens = []
        inds = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        for token in inds:
            if token != last and token != self.EMPTY_TOK:
                tokens.append(token)
            last = token
        #print('inds!!!')
        #print(self.decode(tokens))
        return self.decode(tokens)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
