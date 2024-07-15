import re
from typing import Optional

PGN_VOCAB = [" "]
PGN_VOCAB += [chr(i) for i in range(48, 58)]  # 0-9
PGN_VOCAB += [chr(i) for i in range(97, 105)]  # a-h
PGN_VOCAB += ["B", "K", "N", "Q", "R"]  # pieces
PGN_VOCAB += [f"{x}." for x in range(1, 51)]  # start of turn (up to 50 moves)
PGN_VOCAB += ["O-O", "O-O-O"]  # castling
PGN_VOCAB += [" ", "x", "+", "#"]  # capture / check(mate)
PGN_VOCAB += ["0-1", "1-0", "1/2-1/2"]  # outcome


class PGNTokenizer:
    def __init__(self, vocab: Optional[list[str]] = None):
        self.vocab = PGN_VOCAB if vocab is None else vocab
        self.word_ids = {word: i for i, word in enumerate(self.vocab)}

        re_special_chars = re.compile(r"([\+\*\?\^\$\\\.\[\]\{\}\(\)\|\/])")
        re_words = sorted(re_special_chars.sub(r"\\\1", word) for word in self.vocab)
        self.re_vocab = re.compile("(" + "|".join(re_words[::-1]) + ")")

    def encode(self, seq: str) -> list[int]:
        token_ids: list[int] = []
        while seq:
            if not (match := self.re_vocab.match(seq)):
                raise ValueError(f"could not find next token for '{seq}'")
            word = match.group(0)
            token_ids.append(self.word_ids[word])
            seq = seq[len(word) :]
        return token_ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)
