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

        # state for cluster extraction
        self._dimension: int = 0
        self._sorted_values: List[float] = []
        self._sorted_labels: List[str] = []
        self._tolerance: float = 0.0
        self._biggest_cluster: List[str] = []
        self._current_cluster: List[str] = []
        self._current_start_index: int = 0
        self._current_end_index: int = 0

    def clusters(self) -> List[Cluster]:
        return self._clusters

    def run(self) -> None:
        try:
            while not self._work.empty():
                self._dimension = self._work.get_nowait()
                logging.info("working on dimension %d", self._dimension)

                cluster: Cluster = self._extract_cluster()

                logging.info("%s", cluster)

                self._clusters.append(cluster)

        except Empty:
            pass

        logging.info("no more work to do. stopping thread", self.name)

    def _extract_cluster(self) -> Cluster:
        cluster: Cluster = Cluster(self._dimension, self._dimension)

        vector_values: List[float] = [vector[self._dimension].item() for vector in self._embedding.vectors]
        vector_labels: List[str] = list(self._embedding.vocab.keys())

        sorted_tuples: List[Tuple[float, str]] = sorted(zip(vector_values, vector_labels), key=lambda x: x[0])

        self._sorted_values = [x[0] for x in sorted_tuples]
        self._sorted_labels = [x[1] for x in sorted_tuples]
        self._create_biggest_cluster()

        for label in self._biggest_cluster:
            cluster.add(label)

        return cluster

    def _create_biggest_cluster(self) -> None:
        self._tolerance = statistics.mean(self._sorted_values)
        self._biggest_cluster.clear()
        self._current_start_index = 0

        start_time: float = time.perf_counter()
        create_current_execution_times: List[float] = []
        find_next_execution_times: List[float] = []

        while self._start() < len(self._sorted_values):
            create_current_execution_times.append(SimilarDimensionClusterWorker._profile(self._create_current_cluster))
            find_next_execution_times.append(SimilarDimensionClusterWorker._profile(self._find_next_start_index))

            if len(self._current_cluster) > len(self._biggest_cluster):
                self._biggest_cluster = self._current_cluster

            if len(create_current_execution_times) % 5000 == 0:
                progression: float = self._current_start_index / len(self._sorted_values)
                logging.info(f"[DIMENSION-{self._dimension}]\t{'%06.2f%%' % (progression * 100.0)}")
                logging.info(f"[DIMENSION-{self._dimension}]\t\t{'%06.4fs' % (time.perf_counter() - start_time)}")
                logging.info(f"[DIMENSION-{self._dimension}]\t\t<<_create_current_cluster>>: "
                             f"avg: {'%06.4fs' % statistics.mean(create_current_execution_times)}; "
                             f"tot: {'%06.4fs' % sum(create_current_execution_times)}")
                logging.info(f"[DIMENSION-{self._dimension}]\t\t<<_find_next_start_index>>: "
                             f"avg: {'%06.4fs' % statistics.mean(find_next_execution_times)}; "
                             f"tot: {'%06.4fs' % sum(find_next_execution_times)}")

                create_current_execution_times.clear()
                find_next_execution_times.clear()
                start_time = time.perf_counter()

    @staticmethod
    def _profile(method: FunctionType, *arguments: Any) -> float:
        start_time: float = time.perf_counter()
        method(*arguments)
        end_time: float = time.perf_counter()

        return end_time - start_time

    def _create_current_cluster(self) -> None:
        self._current_cluster = [self._sorted_labels[self._current_start_index]]
        for self._current_end_index in range(self._current_start_index + 1, len(self._sorted_values)):

            if self._end() - self._start() > self._tolerance:
                break

            self._current_cluster.append(self._sorted_labels[self._current_end_index])

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
