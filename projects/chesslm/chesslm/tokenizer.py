import re
from functools import cached_property
from typing import Optional

import pygtrie

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
PGN_VOCAB: list[str] = [PAD_TOKEN, UNK_TOKEN]

UNK_TOKEN_ID = PGN_VOCAB.index(UNK_TOKEN)
PAD_TOKEN_ID = PGN_VOCAB.index(PAD_TOKEN)

START_TOKEN = "1."  # game start
PGN_VOCAB += [START_TOKEN]
PGN_VOCAB += [f"{x}." for x in range(2, 10)]  # start of turn (2-10 moves)
PGN_VOCAB += [f"{x}." for x in range(11, 51)]  # start of turn (up to 50 moves)
PGN_VOCAB += ["0."]  # start of turn catchall (60, 70, etc.)

END_TOKENS = ["0-1", "1-0", "1/2-1/2"]  # outcome
PGN_VOCAB += END_TOKENS

START_TOKEN_ID = PGN_VOCAB.index(START_TOKEN)
END_TOKEN_IDS = [PGN_VOCAB.index(tok) for tok in END_TOKENS]

PGN_VOCAB += [" "]  # spaces
PGN_VOCAB += [str(i) for i in range(10)]  # 0-9
PGN_VOCAB += [chr(i) for i in range(97, 105)]  # a-h
PGN_VOCAB += ["B", "K", "N", "Q", "R"]  # pieces
PGN_VOCAB += ["O-O", "O-O-O", "="]  # castling / promotion
PGN_VOCAB += [" ", "x", "+", "#"]  # capture / check(mate)


class PGNTokenizer:
    def __init__(self, vocab: Optional[list[str]] = None):
        self.vocab = (PGN_VOCAB if vocab is None else vocab).copy()
        self.word_ids = {word: i for i, word in enumerate(self.vocab)}

    @cached_property
    def _re(self) -> re.Pattern:
        # treat special characters so they make sense
        re_special_chars = re.compile(r"([\+\*\?\^\$\\\.\[\]\{\}\(\)\|\/])")
        vocab = [re_special_chars.sub(r"\\\1", word) for word in self.vocab]

        # sort the vocab by length so longest words are prioritized first
        re_words = sorted(vocab)[::-1]

        # combine words with catchall match at the end and compile
        re_unk = r"|[^ ]+"
        return re.compile("(" + "|".join(re_words) + re_unk + ")")

    def findall_encode(self, text: str) -> list[int]:
        return [self.word_ids.get(w, UNK_TOKEN_ID) for w in self._re.findall(text)]

    def encode(self, text: str) -> list[int]:
        return self.findall_encode(text)

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    def validate_encode(self, text: str) -> list[int]:
        tokens = self.findall_encode(text)
        assert tokens[0] == START_TOKEN_ID, f"invalid BOS: {text[:20]}"
        assert tokens[-1] in END_TOKEN_IDS, f"invalid EOS: {text[-20:]}"
        return tokens


class PGNTestTokenizer(PGNTokenizer):
    def __init__(self, vocab: Optional[list[str]] = None):
        super().__init__(vocab)

    @cached_property
    def _re_vocab_only(self) -> re.Pattern:
        re_special_chars = re.compile(r"([\+\*\?\^\$\\\.\[\]\{\}\(\)\|\/])")
        re_words = sorted(re_special_chars.sub(r"\\\1", word) for word in self.vocab)
        return re.compile("(" + "|".join(re_words[::-1]) + ")")

    def match_encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        while text:
            if not (match := self._re_vocab_only.match(text)):
                raise ValueError(f"could not find next token for '{text}'")
            token_ids.append(self.word_ids[match.group()])
            text = text[match.end() :]
        return token_ids

    def finditer_encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        end_ = 0
        for match in self._re_vocab_only.finditer(text):
            start, end = match.span()
            if start != end_:
                raise ValueError(
                    f"could not find next token for '{text[end_ : start]}'"
                )
            end_ = end
            token_ids.append(self.word_ids[match.group()])
        return token_ids

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
