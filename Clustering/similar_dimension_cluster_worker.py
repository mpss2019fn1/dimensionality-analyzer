import logging
import statistics
import time
from multiprocessing import Process
from queue import Queue, Empty
from types import FunctionType
from typing import List, Tuple, Any

from gensim.models import KeyedVectors

from Clustering.cluster import Cluster


class SimilarDimensionClusterWorker(Process):

    def __init__(self, name: str, embedding: KeyedVectors, work: Queue):
        super(SimilarDimensionClusterWorker, self).__init__(name=name)
        self._embedding: KeyedVectors = embedding
        self._work: Queue = work
        self._clusters: List[Cluster] = []
        self._minimum_cluster_size: int = max(10, len(self._embedding.vectors) // (self._embedding.vector_size * 10))

        # state for cluster extraction
        self._dimension: int = 0
        self._sorted_values: List[float] = []
        self._sorted_labels: List[str] = []
        self._tolerance: float = 0.0
        self._biggest_cluster: Tuple[int, int] = (0, 0)
        self._current_cluster: Tuple[int, int] = (0, 0)
        self._current_start_index: int = 0
        self._current_end_index: int = 0

    def clusters(self) -> List[Cluster]:
        return self._clusters

    def run(self) -> None:
        try:
            while not self._work.empty():
                self._dimension = self._work.get_nowait()
                logging.info(f"[DIMENSION-{self._dimension}] begin")

                cluster: Cluster = self._extract_cluster()
                if len(cluster) < self._minimum_cluster_size:
                    continue

                logging.info(f"[DIMENSION-{self._dimension}] => cluster.len: {len(cluster)}")
                self._clusters.append(cluster)

        except Empty:
            pass

        logging.info("no more work to do. stopping process", self.name)

    def _extract_cluster(self) -> Cluster:
        cluster: Cluster = Cluster(self._dimension, self._dimension)

        vector_values: List[float] = [vector[self._dimension].item() for vector in self._embedding.vectors]
        vector_labels: List[str] = list(self._embedding.vocab.keys())

        sorted_tuples: List[Tuple[float, str]] = sorted(zip(vector_values, vector_labels), key=lambda x: x[0])

        self._sorted_values = [x[0] for x in sorted_tuples]
        self._sorted_labels = [x[1] for x in sorted_tuples]
        self._tolerance = statistics.mean(self._sorted_values)
        self._create_biggest_cluster()

        logging.info(f"[DIMENSION-{self._dimension}] done")

        if self._len(self._biggest_cluster) < self._minimum_cluster_size:
            return cluster

        for label_index in range(self._biggest_cluster[0], self._biggest_cluster[1]):
            cluster.add(self._sorted_labels[label_index])

        return cluster

    def _create_biggest_cluster(self) -> None:
        self._biggest_cluster = (0, 0)
        self._current_start_index = 0

        execution_times: List[float] = []
        log_interval: int = len(self._sorted_values) // 100

        while self._current_start_index < len(self._sorted_values) - 1:
            start_time: float = time.perf_counter()
            self._create_current_cluster()

            if self._len(self._current_cluster) > self._len(self._biggest_cluster):
                self._biggest_cluster = self._current_cluster

            execution_times.append(time.perf_counter() - start_time)

            if len(execution_times) % log_interval == 0:
                progression: float = self._current_start_index / len(self._sorted_values)
                total_time: float = sum(execution_times)
                logging.info(f"[DIMENSION-{self._dimension}]\t{'%06.2f%%' % (progression * 100.0)} "
                             f"avg: {'%06.4fs' % (total_time / len(execution_times))}; "
                             f"tot: {'%06.4fs' % total_time}; "
                             f"<<_biggest_cluster.len>>: {self._len(self._biggest_cluster)}")

                execution_times.clear()

    def _create_current_cluster(self) -> None:
        self._current_cluster = (0, 0)
        self._current_end_index = min(len(self._sorted_values) - 1,
                                      self._current_start_index + self._min_len())

        if self._end() - self._start() > self._tolerance:
            self._current_start_index += 1
            return

        while self._current_end_index < len(self._sorted_values) - 1:
            self._current_end_index = min(len(self._sorted_values) - 1,
                                          self._current_end_index + self._min_len())

            difference: float = self._end() - self._start()

            if difference < self._tolerance:
                continue

            if difference == self._tolerance:
                while difference == self._tolerance and self._current_end_index < len(self._sorted_values) - 1:
                    self._current_end_index += min(len(self._sorted_values) - 1,
                                                   self._current_end_index + 1)
                    difference = self._end() - self._start()

            else:  # difference > self._tolerance
                while difference > self._tolerance:
                    self._current_end_index -= 1
                    difference = self._end() - self._start()

                self._current_end_index += 1  # add one, because the end_index is excluded from this cluster

            break

        self._current_cluster = (self._current_start_index, self._current_end_index)
        self._find_next_start_index()

    def _find_next_start_index(self) -> None:
        if self._current_end_index - self._current_start_index < 2:
            self._current_start_index = self._current_end_index
            return

        next_start_minus_one: int = 0
        for next_start_minus_one in range(self._current_end_index - 1, -1, -1):
            if self._end() - self._sorted_values[next_start_minus_one] > self._tolerance:
                break

        self._current_start_index = next_start_minus_one + 1

    def _start(self) -> float:
        return self._sorted_values[self._current_start_index]

    def _end(self) -> float:
        return self._sorted_values[self._current_end_index]

    @staticmethod
    def _len(cluster: Tuple[int, int]) -> int:
        return cluster[1] - cluster[0]

    def _min_len(self) -> int:
        return max(self._minimum_cluster_size, self._len(self._biggest_cluster))
