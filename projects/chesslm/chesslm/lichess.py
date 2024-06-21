import io
import re
from typing import Callable, Iterator, Optional, TypeVar

from pydantic import BaseModel
from tqdm import tqdm
from zstandard import ZstdDecompressor

RE_ENTRY_METADATA = re.compile(r'\[(\w+) "(.+)"\]\n?')
RE_SEQUENCE_TURN_METADATA = re.compile(r" \{ \[[^\]]+\] \} ")
RE_SEQUENCE_TURN_CONTINUATION = re.compile(r" \d+\.{3}")
RE_LICHESS_TOURNAMENT = re.compile(r" https://lichess.org/tournament/\w+")

T = TypeVar("T")


class LichessEntry(BaseModel):
    event: str  # [Event "Rated Classical game"]
    url: str  # [Site "https://lichess.org/j1dkb5dw"]
    result: str  # [Result "1-0"]
    utc_timestamp: str  # [UTCDate "2012.12.31"] [UTCTime "23:04:57"]
    white_elo: Optional[int]  # [WhiteElo "1824"]
    black_elo: Optional[int]  # [BlackElo "1973"]
    time_control: str  # [TimeControl "60+1"]
    termination: str  # [Termination "Normal"]
    sequence: str  # 1. e4 e6 2. d4 b6 3. a3 Bb7 ... 13. Qe8# 1-0

    @property
    def site(self) -> str:
        return f"https://lichess.org/{self.url}"

    @property
    def elo(self) -> Optional[int]:
        if (self.white_elo is None) or (self.black_elo is None):
            return None
        return (self.white_elo + self.black_elo) // 2

    @property
    def sequence_has_eval(self) -> bool:
        return "%eval" in self.sequence

    @property
    def sequence_has_metadata(self) -> bool:
        return " { [%" in self.sequence

    @property
    def plain_sequence(self) -> str:
        if self.sequence_has_metadata:
            sequence = RE_SEQUENCE_TURN_METADATA.sub(" ", self.sequence)
            return RE_SEQUENCE_TURN_CONTINUATION.sub(" ", sequence)
        return self.sequence

    @property
    def ends_with_checkmate(self) -> bool:
        return self.plain_sequence.endswith(f"# {self.result}")


class LichessPGNReader:
    def __init__(self, filepath: str, n_total: Optional[int] = None):
        self.filepath = filepath
        self.n_total = n_total
        self.reset()

    def get_stream(self) -> io.TextIOWrapper:
        if self.filepath.endswith(".pgn"):
            return open(self.filepath, "r")
        elif self.filepath.endswith(".pgn.zst"):
            f = open(self.filepath, "rb")
            dctx = ZstdDecompressor()
            reader = dctx.stream_reader(f)
            return io.TextIOWrapper(reader, encoding="utf-8")
        else:
            raise ValueError(f"{self.filepath} is neither a .pgn or .pgn.zst file")

    def reset(self):
        self.stream: io.TextIOWrapper = self.get_stream()
        self.lastlines: list[str] = []

    @property
    def nextline(self) -> str:
        return self.stream.readline()

    @staticmethod
    def parse_metadata(line: str) -> tuple[str, str]:
        if not (match := RE_ENTRY_METADATA.match(line)):
            raise ValueError(f"could not parse metadata from '{line}'")
        key, value = match.group(1), match.group(2)
        return key, value

    @property
    def nextdict(self) -> dict[str, str]:
        d: dict[str, str] = {}
        self.lastlines: list[str] = []

        # metadata or blank lines
        while (line := self.nextline) and line.startswith("["):
            self.lastlines.append(line)
            key, value = self.parse_metadata(line)
            if key in d:
                raise ValueError(f"repeated metadata '{key}'")
            else:
                d[key] = value

        # if no lines read, return empty dict
        if not self.lastlines:
            return {}

        assert line == "\n", "line after metadata should be a blank line"
        self.lastlines.append(line)

        # start of move sequence
        if line := self.nextline:
            self.lastlines.append(line)
            d["_sequence"] = line
        else:
            raise ValueError("move sequence not found")

        # newer data will have several lines for the move sequence
        while len((line := self.nextline).removesuffix("\n")):
            self.lastlines.append(line)
            d["_sequence"] += line

        # don't need any newlines in the move sequence
        d["_sequence"] = d["_sequence"].replace("\n", "")

        assert line == "\n", "final line should be a blank line"
        self.lastlines.append(line)

        return d

    @property
    def nextentry(self) -> LichessEntry:
        if not (d := self.nextdict):
            raise ValueError("no entries remaining")
        white_elo = d["WhiteElo"]
        black_elo = d["BlackElo"]
        return LichessEntry(
            event=RE_LICHESS_TOURNAMENT.sub("", d["Event"]),
            url=d["Site"].split("/")[-1],
            result=d["Result"],
            utc_timestamp=d["UTCDate"] + " " + d["UTCTime"],
            white_elo=int(white_elo) if white_elo.isnumeric() else None,
            black_elo=int(black_elo) if black_elo.isnumeric() else None,
            time_control=d["TimeControl"],
            termination=d["Termination"],
            sequence=d["_sequence"],
        )

    def __iter__(self) -> "LichessPGNReader":
        self.reset()
        return self

    def __next__(self) -> LichessEntry:
        try:
            return self.nextentry
        except ValueError:
            raise StopIteration

    def create_tqdm(self, limit: Optional[int], desc: str) -> tqdm:
        if isinstance(limit, int):
            total = limit
        elif isinstance(self.n_total, int):
            total = self.n_total
        else:
            total = None
        return tqdm(total=total, desc=desc, unit_scale=True, unit_divisor=1000)

    def tqdm_count(
        self,
        funcs: dict[str, Callable[[LichessEntry], T]],
        limit: Optional[int] = None,
        desc: str = "",
    ) -> dict[str, dict[T, int]]:
        """Count entries by iterating over entries with a progress bar

        e.g. reader.tqdm_counts({"event": lambda entry: entry.event})
            {
                'event': {
                    'Rated Classical game': 41772,
                    'Rated Bullet game': 32691,
                    'Rated Blitz game': 45388,
                    'Rated Correspondence game': 266,
                    'Rated Blitz tournament': 896,
                    'Rated Bullet tournament': 295,
                    'Rated Classical tournament': 24
                }
            }

        Args:
            funcs (dict[str, Callable[[LichessEntry], T]]): a dictionary of functions
                that take a LichessEntry and return a hashable that can be counted.

        Returns:
            dict[str, dict[T, int]]: the counts of the values returned by each function
        """

        has_limit = isinstance(limit, int)
        counts: dict[str, dict[T, int]] = {key: {} for key in funcs}

        with self.create_tqdm(limit, desc) as pbar:
            for i, entry in enumerate(self):
                if has_limit and i >= limit:
                    break
                pbar.update(1)
                for key, func in funcs.items():
                    func_value = func(entry)
                    counts[key][func_value] = counts[key].get(func_value, 0) + 1

        return counts

    def tqdm_map(
        self,
        func: Callable[[LichessEntry], T],
        limit: Optional[int] = None,
        desc: str = "",
    ) -> Iterator[T]:

        has_limit = isinstance(limit, int)
        with self.create_tqdm(limit, desc) as pbar:
            for i, entry in enumerate(self):
                if has_limit and i >= limit:
                    break
                pbar.update(1)
                yield func(entry)

    def tqdm_filter(
        self,
        func: Callable[[LichessEntry], bool],
        limit: Optional[int] = None,
        desc: str = "Retrieved {n} entries",
    ) -> Iterator[LichessEntry]:

        has_limit = isinstance(limit, int)
        counter = 0
        with self.create_tqdm(limit, desc.format(n=0)) as pbar:
            for i, entry in enumerate(self):
                if has_limit and i >= limit:
                    break
                pbar.update(1)
                if func(entry):
                    counter += 1
                    if counter < 1_000_000:
                        if counter % 1500 == 0:
                            n = f"{counter // 1000}k"
                            pbar.set_description(desc.format(n=n))
                    elif counter % 100_000 == 0:
                        n = f"{(counter // 100_000) / 10}M"
                        pbar.set_description(desc.format(n=n))
                    yield entry
            pbar.set_description(desc.format(n="{:,}".format(counter)))
