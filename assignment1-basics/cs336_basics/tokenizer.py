from typing import Iterable, Iterator

import regex as re


class Tokenizer:

    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
    ):
        self.special_tokens = special_tokens if special_tokens else []
        if self.special_tokens:
            self.special_tokens.sort(key=lambda x: len(x), reverse=True)
        self.vocab = vocab
        self.rvocab = {v: k for k, v in vocab.items()}
        for token in self.special_tokens:
            if token.encode("utf-8") not in self.rvocab:
                self.vocab[len(self.vocab)] = token.encode('utf-8')
                self.rvocab[token.encode("utf-8")] = len(self.vocab) - 1
        self.merges = merges

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
    ):
        raise NotImplementedError

    def _encode_subword(self, subword: str) -> list[int]:
        sb = tuple([bytes([o]) for o in list(subword.encode("utf-8"))])

        i = 0
        while i < len(self.merges):
            j = 0
            if len(sb) == 1:
                return [self.rvocab[sb[0]]]
            while j + 1 < len(sb):
                if self.merges[i] == (sb[j], sb[j + 1]):
                    sb = (*sb[:j], sb[j] + sb[j + 1], *sb[j + 2:])
                    i = 0
                    break
                j += 1
            else:
                i += 1
        return [self.rvocab[o] for o in sb]

    def encode(
            self,
            text: str
    ) -> list[int]:
        parts = []
        if self.special_tokens:
            escaped_tokens = [re.escape(o) for o in self.special_tokens]
            special_tok_pat = '(' + '|'.join(escaped_tokens) + ')'
            parts = re.split(special_tok_pat, text)
        else:
            parts.append(text)

        token_ids = []
        for part in parts:
            if part in self.special_tokens:
                token_ids.append(self.rvocab[part.encode('utf-8')])
                continue
            subwords = []
            for subword in re.finditer(self.PAT, part, re.IGNORECASE): subwords.append(subword.captures()[0])
            for subword in subwords: token_ids.extend(self._encode_subword(subword))
        return token_ids

    def encode_iterable(
            self,
            iterable: Iterable[str]
    ) -> Iterator[int]:
        special_tok_pat = None
        if self.special_tokens:
            escaped_tokens = [re.escape(o) for o in self.special_tokens]
            special_tok_pat = '(' + '|'.join(escaped_tokens) + ')'

        for chunk in iterable:
            if self.special_tokens:
                parts = re.split(special_tok_pat, chunk)
                for i, part in enumerate(parts):
                    if part in self.special_tokens:
                        yield self.rvocab[part.encode('utf-8')]
                    else:
                        for subword in re.finditer(self.PAT, part, re.IGNORECASE):
                            yield from self._encode_subword(subword.captures()[0])
            else:
                for subword in re.finditer(self.PAT, chunk, re.IGNORECASE):
                    yield from self._encode_subword(subword.captures()[0])

    def decode(
            self,
            ids: list[int]
    ) -> str:
        return bytes.decode(b''.join([self.vocab[id] for id in ids]), errors="replace")
