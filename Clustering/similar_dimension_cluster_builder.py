from queue import Queue
from typing import List

from gensim.models import KeyedVectors

from Clustering.abstract_cluster_builder import AbstractClusterBuilder
from Clustering.cluster import Cluster
from Clustering.similar_dimension_cluster_worker import SimilarDimensionClusterWorker


class SimilarDimensionClusterBuilder(AbstractClusterBuilder):

    def __init__(self, embedding: KeyedVectors, workers: int):
        super(SimilarDimensionClusterBuilder, self).__init__(embedding)
        self._number_of_workers = workers

    def run(self) -> List[Cluster]:
        queues: List[Queue] = [Queue() for _ in range(self._number_of_workers)]
        for dimension in range(self._embedding.vector_size):
            queues[dimension % self._number_of_workers].put_nowait(dimension)

        workers: List[SimilarDimensionClusterWorker] = []
        for i in range(self._number_of_workers):
            worker: SimilarDimensionClusterWorker = SimilarDimensionClusterWorker(name=f"{i}",
                                                                                  embedding=self._embedding,
                                                                                  work=queues[i])
            worker.start()
            workers.append(worker)

        clusters: List[Cluster] = []
        for worker in workers:
            worker.join()
            clusters.extend(worker.clusters())

        return clusters
