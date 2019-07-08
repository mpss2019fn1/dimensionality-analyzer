from abc import ABC, abstractmethod
from typing import List

from gensim.models import KeyedVectors

from Clustering.cluster import Cluster


class AbstractClusterBuilder(ABC):

    def __init__(self, embedding: KeyedVectors):
        self._embedding: KeyedVectors = embedding

    @abstractmethod
    def run(self) -> List[Cluster]:
        raise NotImplementedError
