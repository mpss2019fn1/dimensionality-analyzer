import logging
import statistics
from multiprocessing import Process
from queue import Queue, Empty
from typing import List, Tuple

from gensim.models import KeyedVectors

from Clustering.cluster import Cluster


class SimilarDimensionClusterWorker(Process):

    def __init__(self, name: str, embedding: KeyedVectors, work: Queue):
        super(SimilarDimensionClusterWorker, self).__init__(name=name)
        self._embedding: KeyedVectors = embedding
        self._work: Queue = work
        self._clusters: List[Cluster] = []

    def clusters(self) -> List[Cluster]:
        return self._clusters

    def run(self) -> None:
        try:
            while not self._work.empty():
                dimension: int = self._work.get_nowait()
                logging.info("working on dimension %d", dimension)

                cluster: Cluster = self._extract_cluster(dimension)

                logging.info("%s", cluster)

                self._clusters.append(cluster)

        except Empty:
            pass

        logging.info("no more work to do. stopping thread", self.name)

    def _extract_cluster(self, dimension: int) -> Cluster:
        cluster: Cluster = Cluster(dimension, dimension)

        vector_values: List[float] = [vector[dimension].item() for vector in self._embedding.vectors]
        vector_labels: List[str] = list(self._embedding.vocab.keys())

        for label in self._extract_closest_labels(list(zip(vector_labels, vector_values))):
            cluster.add(label)

        return cluster

    @staticmethod
    def _extract_closest_labels(values: List[Tuple[str, float]]) -> List[str]:
        sorted_values: List[Tuple[str, float]] = sorted(values, key=lambda x: x[1])
        tolerance: float = statistics.mean([x[1] for x in sorted_values])
        largest_cluster: List[str] = []

        for start_index in range(len(sorted_values)):
            start_label: str = sorted_values[start_index][0]
            start_value: float = sorted_values[start_index][1]
            current_cluster: List[str] = [start_label]

            for end_index in range(start_index + 1, len(sorted_values)):
                current_label: str = sorted_values[end_index][0]
                current_value: float = sorted_values[end_index][1]

                if current_value - start_value > tolerance:
                    break

                current_cluster.append(current_label)

            if len(current_cluster) > len(largest_cluster):
                largest_cluster = current_cluster

        return largest_cluster
