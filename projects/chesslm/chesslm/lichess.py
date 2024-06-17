import io
import os
import re
from typing import Callable, Optional, TypeVar

from pydantic import BaseModel
from tqdm import tqdm

RE_PGN_METADATA = re.compile(r'\[(\w+) "(.+)"\]\n?')
RE_LICHESS_TOURNAMENT = re.compile(r" https://lichess.org/tournament/\w+")

T = TypeVar("T")


class LichessEntry(BaseModel):
    event: str  # [Event "Rated Bullet game"]
    url: str  # [Site "https://lichess.org/rklpc7mk"]
    result: str  # [Result "0-1"]
    utc_timestamp: str  # [UTCDate "2012.12.31"] [UTCTime "23:04:57"]
    white_elo: str  # [WhiteElo "1824"]
    black_elo: str  # [BlackElo "1973"]
    time_control: str  # [TimeControl "60+1"]
    termination: str  # [Termination "Normal"]
    sequence: str  # 1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 ... 11. Nxc7+ 1-0

    @property
    def site(self) -> str:
        return f"https://lichess.org/{self.url}"

    @property
    def elo(self) -> float:
        if self.white_elo.isnumeric() and self.black_elo.isnumeric():
            return (int(self.white_elo) + int(self.black_elo)) / 2
        else:
            return float("nan")


class LichessPGNReader:
    def __init__(self, filepath: str):
        self.file: io.TextIOWrapper = open(filepath, "r")
        self.filesize: int = os.path.getsize(filepath)
        self.reset()

    def reset(self):
        self.file.seek(0)
        self.bytes_read: int = 0
        self.last_bytes_read: int = 0
        self.lastlines: list[str] = []

    @property
    def nextline(self) -> str:
        line = self.file.readline()
        self.bytes_read += len(line)
        return line

    @staticmethod
    def parse_metadata(line: str) -> tuple[str, str]:
        if not (match := RE_PGN_METADATA.match(line)):
            raise ValueError(f"could not parse metadata from '{line}'")
        key, value = match.group(1), match.group(2)
        return key, value

    @property
    def nextdict(self) -> dict[str, str]:
        d: dict[str, str] = {}
        self.last_bytes_read = self.bytes_read
        self.lastlines: list[str] = []

        while (line := self.nextline) and (line.startswith("[")):
            # all of this is metadata or blank
            self.lastlines.append(line)
            key, value = self.parse_metadata(line)
            if key in d:
                raise ValueError(f"repeated metadata '{key}'")
            else:
                d[key] = value

        if not self.lastlines:
            return {}  # no lines read, return empty dict

        if line := self.nextline:
            self.lastlines.append(line)
            d["_sequence"] = line
        else:
            raise ValueError("move sequence not found")

        assert (line := self.nextline) == "\n", "final line should be a blank line"
        self.lastlines.append(line)

        return d

    @property
    def lastbytes(self) -> int:
        return self.bytes_read - self.last_bytes_read

    @property
    def nextentry(self) -> Optional[LichessEntry]:
        if not (d := self.nextdict):
            return None
        return LichessEntry(
            event=RE_LICHESS_TOURNAMENT.sub("", d["Event"]),
            url=d["Site"].split("/")[-1],
            result=d["Result"],
            utc_timestamp=d["UTCDate"] + " " + d["UTCTime"],
            white_elo=d["WhiteElo"],
            black_elo=d["BlackElo"],
            time_control=d["TimeControl"],
            termination=d["Termination"],
            sequence=d["_sequence"],
        )

    def __iter__(self) -> "LichessPGNReader":
        self.reset()
        return self

    def __next__(self) -> LichessEntry:
        if entry := self.nextentry:
            return entry
        else:
            raise StopIteration

    def create_tqdm(self, desc: str = "") -> tqdm:
        return tqdm(
            desc=desc,
            total=self.filesize,
            unit="B",
            unit_scale=True,
            unit_divisor=1000,
        )

    def tqdm_process(self, func: Callable[[LichessEntry], T]) -> list[T]:
        li: list[T] = []
        with self.create_tqdm() as pbar:
            for entry in self:
                pbar.update(self.lastbytes)
                li.append(func(entry))
        return li

    def tqdm_count(
        self, funcs: dict[str, Callable[[LichessEntry], T]]
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

        sets: dict[str, dict[T, int]] = {key: {} for key in funcs}

        with self.create_tqdm() as pbar:
            for entry in self:
                pbar.update(self.lastbytes)
                for key, func in funcs.items():
                    func_value = func(entry)
                    sets[key][func_value] = sets[key].get(func_value, 0) + 1

        return sets
