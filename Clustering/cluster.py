from typing import Set, Iterator


class Cluster:

    def __init__(self, identifier: int, dimension: int):
        self._identifier: int = identifier
        self._dimension: int = dimension
        self._entities: Set[str] = set()

    def add(self, entity: str) -> None:
        self._entities.add(entity)

    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entities)
