import re
from functools import cached_property

import pygtrie

# Start, end, special tokens, spaces, first turn and outcome tokens
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SPACE_TOKEN = " "
FIRST_TURN_TOKEN = "1."
OUTCOME_TOKENS = ["0-1", "1-0", "1/2-1/2"]
PGN_VOCAB: list[str] = [
    START_TOKEN,
    END_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    SPACE_TOKEN,
    FIRST_TURN_TOKEN,
    *OUTCOME_TOKENS,
]
(
    START_TOKEN_ID,
    END_TOKEN_ID,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    SPACE_TOKEN_ID,
    FIRST_TURN_TOKEN_ID,
    *OUTCOME_TOKEN_IDS,
) = tuple(tuple(range(len(PGN_VOCAB))))

# Start of turn
PGN_VOCAB += [f"{x}." for x in range(2, 10)]  # start of turn (2-10 moves)
PGN_VOCAB += [f"{x}." for x in range(11, 51)]  # start of turn (up to 50 moves)
PGN_VOCAB += ["0."]  # start of turn catchall (60, 70, etc.)

# Numbers and letters (for board positions)
PGN_VOCAB += [str(i) for i in range(10)]  # 0-9
PGN_VOCAB += [chr(i) for i in range(97, 105)]  # a-h

# Misc.
PGN_VOCAB += ["B", "K", "N", "Q", "R"]  # pieces
PGN_VOCAB += ["O-O", "O-O-O", "="]  # castling / promotion
PGN_VOCAB += ["x", "+", "#"]  # capture / check(mate)

# Word ids
PGN_IDS = {word: i for i, word in enumerate(PGN_VOCAB)}


class PGNTokenizer:
    @property
    def n_vocab(self) -> int:
        return len(PGN_VOCAB)

    @cached_property
    def _re(self) -> re.Pattern:
        # treat special characters so they make sense
        re_special_chars = re.compile(r"([\+\*\?\^\$\\\.\[\]\{\}\(\)\|\/])")
        vocab = [re_special_chars.sub(r"\\\1", word) for word in PGN_VOCAB]

        # sort the vocab by length so longest words are prioritized first
        re_words = sorted(vocab)[::-1]

        # combine words with catchall match at the end and compile
        re_unk = r"|[^ ]+"
        return re.compile("(" + "|".join(re_words) + re_unk + ")")

    def findall_encode(self, text: str) -> list[int]:
        return [PGN_IDS.get(w, UNK_TOKEN_ID) for w in self._re.findall(text)]

    @staticmethod
    def add_special_tokens(text: str) -> str:
        if not text.startswith(START_TOKEN):
            text = START_TOKEN + text
        if not text.endswith(END_TOKEN):
            text = text + END_TOKEN
        return text

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if add_special_tokens:
            text = self.add_special_tokens(text)
        return self.findall_encode(text)

    def decode(self, ids: list[int]) -> str:
        return "".join(PGN_VOCAB[i] for i in ids)

    def validate_encode(
        self, text: str, add_special_tokens: bool = True, allow_unknowns: bool = False
    ) -> list[int]:
        text = self.add_special_tokens(text)
        tokens = self.encode(text, add_special_tokens=False)

        assert tokens[:2] == [
            START_TOKEN_ID,
            FIRST_TURN_TOKEN_ID,
        ], f"invalid BOS: {text[:20]}"

        assert (tokens[-2] in OUTCOME_TOKEN_IDS) and tokens[
            -1
        ] == END_TOKEN_ID, f"invalid EOS: {text[-20:]}"

        if not allow_unknowns:
            assert UNK_TOKEN_ID not in tokens, "<unk> tokens detected"

        return tokens if add_special_tokens else tokens[1:-1]


class PGNTestTokenizer(PGNTokenizer):
    @cached_property
    def _re_vocab_only(self) -> re.Pattern:
        re_special_chars = re.compile(r"([\+\*\?\^\$\\\.\[\]\{\}\(\)\|\/])")
        re_words = sorted(re_special_chars.sub(r"\\\1", word) for word in PGN_VOCAB)
        return re.compile("(" + "|".join(re_words[::-1]) + ")")

    def match_encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        while text:
            if not (match := self._re_vocab_only.match(text)):
                raise ValueError(f"could not find next token for '{text}'")
            token_ids.append(PGN_IDS[match.group()])
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
            token_ids.append(PGN_IDS[match.group()])
        return token_ids

    @cached_property
    def trie(self) -> pygtrie.CharTrie:
        return pygtrie.CharTrie(PGN_IDS)

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
        return "".join(PGN_VOCAB[i] for i in ids)
