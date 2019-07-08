from abc import ABC
from pathlib import Path


class AbstractFileParser(ABC):

    def __init__(self, file_path: Path):
        self._file_path: Path = file_path
