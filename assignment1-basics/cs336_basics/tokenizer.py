from typing import Iterable, Iterator
import regex as re

class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.rvocab = {v:k for k,v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        raise NotImplementedError

    def _encode_subword(self, subword: str):
        sb = tuple([bytes([o]) for o in list(subword.encode("utf-8"))])

        i = 0
        while len(sb) > 1 and i+1 < len(sb):
            for p in self.merges:
                if (sb[i], sb[i+1]) == p:
                    sb = (*sb[:i], sb[i]+sb[i+1], *sb[i+2:])
                    i = 0 # start searching from the beginning again
                    break
            # if pair is not found in  merges 
            else: i += 1 

        return [self.rvocab[o] for o in sb]

    def encode(
        self,
        text: str
    ) -> list[int]:

        parts = []
        if self.special_tokens:
            escaped_tokens = [re.escape(o) for o in self.special_tokens]
            special_tok_pat = '|'.join(escaped_tokens)
            parts = re.split(special_tok_pat, text)
        else: parts.append(text)

        token_ids = []
        for i,part in enumerate(parts):
            subwords = []
            for subword in re.finditer(self.PAT, part, re.IGNORECASE):
                subwords.append(subword.captures()[0])

            for subword in subwords: token_ids.extend(self._encode_subword(subword))

            if self.special_tokens and i < len(parts)-1:
                token_ids.append(self.rvocab[self.special_tokens[0].encode('utf-8')])
        
        return token_ids
        

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        raise NotImplementedError
    

    def decode(
        self,
        ids: list[int]
    ) -> str:
        return ''.join([bytes.decode(self.vocab[id], errors="replace") for id in ids])
