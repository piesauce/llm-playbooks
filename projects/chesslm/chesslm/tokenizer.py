import re
from functools import cached_property
from typing import Optional

import pygtrie

PGN_VOCAB: list[str] = [" "]  # spaces
PGN_VOCAB += [str(i) for i in range(10)]  # 0-9
PGN_VOCAB += [chr(i) for i in range(97, 105)]  # a-h
PGN_VOCAB += ["B", "K", "N", "Q", "R"]  # pieces
PGN_VOCAB += [f"{x}." for x in range(51)]  # start of turn (up to 50 moves)
PGN_VOCAB += ["O-O", "O-O-O", "="]  # castling / promotion
PGN_VOCAB += [" ", "x", "+", "#"]  # capture / check(mate)
PGN_VOCAB += ["0-1", "1-0", "1/2-1/2"]  # outcome

UNK_TOKEN = "<unk>"
UNK_TOKEN_ID = -1
SPECIAL_TOKEN_IDS = {UNK_TOKEN: UNK_TOKEN_ID}


class PGNTokenizer:
    def __init__(self, vocab: Optional[list[str]] = None):
        self.vocab = (PGN_VOCAB if vocab is None else vocab).copy()
        self.word_ids = {word: i for i, word in enumerate(self.vocab)}
        self.word_ids = SPECIAL_TOKEN_IDS | self.word_ids

    @cached_property
    def _re(self) -> re.Pattern:
        re_special_chars = re.compile(r"([\+\*\?\^\$\\\.\[\]\{\}\(\)\|\/])")
        re_words = sorted(re_special_chars.sub(r"\\\1", word) for word in self.vocab)
        return re.compile("(" + "|".join(re_words[::-1]) + r"|[^ ]+" ")")

    def encode(self, text: str) -> list[int]:
        return [self.word_ids.get(w, UNK_TOKEN_ID) for w in self._re.findall(text)]

    @cached_property
    def trie(self) -> pygtrie.CharTrie:
        return pygtrie.CharTrie(self.word_ids)

    def trie_encode(self, text: str) -> list[int]:
        # A "ground truth" longest-match encoder. Very slow, only use for debugging.
        token_ids: list[int] = []
        while text:
            word, i = self.trie.longest_prefix(text)
            if not word:
                raise ValueError(f"could not find next token for '{text}'")
            token_ids.append(i)
            text = text[len(word) :]
        return token_ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)
